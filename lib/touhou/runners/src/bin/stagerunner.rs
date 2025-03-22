use luminance::blending::{Equation, Factor};
use luminance::context::GraphicsContext;
use luminance::pipeline::{BoundTexture, PipelineState};
use luminance::pixel::NormUnsigned;
use luminance::render_state::RenderState;
use luminance::shader::program::{Program, Uniform};
use luminance::tess::{Mode, TessBuilder};
use luminance::texture::Dim2Array;
use luminance_derive::{Semantics, Vertex, UniformInterface};
use luminance_glfw::{Action, Key, WindowEvent, GlfwSurface, Surface, WindowDim, WindowOpt};
use touhou_formats::th06::anm0::Anm0;
use touhou_formats::th06::ecl::{Ecl, Rank, MainInstruction};
use touhou_interpreters::th06::anm0::Vertex as FakeVertex;
use touhou_interpreters::th06::ecl::EclRunner;
use touhou_interpreters::th06::enemy::{Enemy, Game, Position};
use touhou_utils::math::{perspective, setup_camera};
use touhou_utils::prng::Prng;
use std::cell::RefCell;
use std::rc::Rc;
use std::env;
use std::path::Path;

use touhou_runners::common::{load_file_into_vec, load_multiple_anm_images, LoadedTexture};

const VS: &str = r#"
in ivec3 in_position;
in uint in_layer;
in vec2 in_texcoord;
in uvec4 in_color;

uniform mat4 mvp;

flat out uint layer;
out vec2 texcoord;
out vec4 color;

void main()
{
    gl_Position = mvp * vec4(vec3(in_position), 1.0);
    texcoord = vec2(in_texcoord);

    // Normalized from the u8 being passed.
    color = vec4(in_color) / 255.;

    layer = in_layer;
}
"#;

const FS: &str = r#"
flat in uint layer;
in vec2 texcoord;
in vec4 color;

uniform sampler2DArray color_map;

out vec4 frag_color;

void main()
{
    frag_color = texture(color_map, vec3(texcoord, layer)) * color;
}
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Semantics)]
pub enum Semantics {
    #[sem(name = "in_position", repr = "[i16; 3]", wrapper = "VertexPosition")]
    Position,

    #[sem(name = "in_layer", repr = "u16", wrapper = "VertexLayer")]
    Layer,

    #[sem(name = "in_texcoord", repr = "[f32; 2]", wrapper = "VertexTexcoord")]
    Texcoord,

    #[sem(name = "in_color", repr = "[u8; 4]", wrapper = "VertexColor")]
    Color,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Vertex)]
#[vertex(sem = "Semantics")]
struct Vertex {
    pos: VertexPosition,
    layer: VertexLayer,
    uv: VertexTexcoord,
    rgba: VertexColor,
}

#[derive(UniformInterface)]
struct ShaderInterface {
    // the 'static lifetime acts as “anything” here
    color_map: Uniform<&'static BoundTexture<'static, Dim2Array, NormUnsigned>>,

    #[uniform(name = "mvp")]
    mvp: Uniform<[[f32; 4]; 4]>,
}

fn main() {
    // Parse arguments.
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <unarchived ST.DAT directory> <stage number> <easy|normal|hard|lunatic>", args[0]);
        return;
    }
    let directory = Path::new(&args[1]);
    let stage_number: u8 = args[2].parse().expect("stage");
    let rank: Rank = args[3].parse().expect("rank");

    // Open the ECL file.
    let buf = load_file_into_vec(directory.join(format!("ecldata{}.ecl", stage_number))).unwrap();
    let (_, ecl) = Ecl::from_slice(&buf).unwrap();
    assert_eq!(ecl.mains.len(), 1);
    let main = ecl.mains[0].clone();

    // Open the ANM file.
    let anm_filename = directory.join(format!("stg{}enm.anm", stage_number));
    let buf = load_file_into_vec(&anm_filename).unwrap();
    let (_, mut anms) = Anm0::from_slice(&buf).unwrap();
    let anm0 = anms.pop().unwrap();

    // Open the second ANM file.
    let anm2_filename = directory.join(format!("stg{}enm2.anm", stage_number));
    let buf = load_file_into_vec(&anm2_filename).unwrap();
    let (_, mut anms) = Anm0::from_slice(&buf).unwrap();
    let anm0_bis = anms.pop().unwrap();

    let anms = [anm0, anm0_bis];

    // Get the time since January 1970 as a seed for the PRNG.
    let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let prng = Rc::new(RefCell::new(Prng::new(time.subsec_micros() as u16)));

    // Create the Game god object.
    let game = Game::new(prng, rank);
    let game = Rc::new(RefCell::new(game));

    assert_eq!(std::mem::size_of::<Vertex>(), std::mem::size_of::<FakeVertex>());
    let vertices: [Vertex; 4] = {
        let data = std::mem::MaybeUninit::uninit();
        unsafe { data.assume_init() }
    };

    let mut surface = GlfwSurface::new(WindowDim::Windowed(384, 448), "Touhou", WindowOpt::default()).unwrap();

    // Open the image atlas matching this ANM.
    let tex = load_multiple_anm_images(&mut surface, &anms, &anm_filename).expect("image loading");
    let anms = Rc::new(RefCell::new(anms));

    // set the uniform interface to our type so that we can read textures from the shader
    let program =
        Program::<Semantics, (), ShaderInterface>::from_strings(None, VS, None, FS).expect("program creation").ignore_warnings();

    let mut tess = TessBuilder::new(&mut surface)
        .add_vertices(vertices)
        .set_mode(Mode::TriangleFan)
        .build()
        .unwrap();

    let mut back_buffer = surface.back_buffer().unwrap();
    let mut resize = false;
    let mut frame = 0;
    let mut ecl_runners = vec![];

    'app: loop {
        for event in surface.poll_events() {
            match event {
                WindowEvent::Close | WindowEvent::Key(Key::Escape, _, Action::Release, _) => break 'app,

                WindowEvent::FramebufferSize(..) => {
                    resize = true;
                }

                _ => (),
            }
        }

        if resize {
            back_buffer = surface.back_buffer().unwrap();
            resize = false;
        }

        for call in main.instructions.iter() {
            if call.time == frame {
                let sub = call.sub;
                let instr = call.instr;
                let (x, y, _z, life, bonus, score, mirror) = match instr {
                    MainInstruction::SpawnEnemy(x, y, z, life, bonus, score) => (x, y, z, life, bonus, score, false),
                    MainInstruction::SpawnEnemyMirrored(x, y, z, life, bonus, score) => (x, y, z, life, bonus, score, true),
                    MainInstruction::SpawnEnemyRandom(x, y, z, life, bonus, score) => (x, y, z, life, bonus, score, false),
                    MainInstruction::SpawnEnemyMirroredRandom(x, y, z, life, bonus, score) => (x, y, z, life, bonus, score, true),
                    _ => continue,
                };
                let enemy = Enemy::new(Position::new(x, y), life, bonus, score, mirror, Rc::downgrade(&anms), Rc::downgrade(&game));
                let runner = EclRunner::new(&ecl, enemy, sub);
                ecl_runners.push(runner);
            }
        }

        for runner in ecl_runners.iter_mut() {
            runner.run_frame();
            let mut enemy = runner.enemy.borrow_mut();
            enemy.update();
        }

        // here, we need to bind the pipeline variable; it will enable us to bind the texture to the GPU
        // and use it in the shader
        surface
            .pipeline_builder()
            .pipeline(&back_buffer, &PipelineState::default(), |pipeline, mut shd_gate| {
                // bind our fancy texture to the GPU: it gives us a bound texture we can use with the shader
                let bound_tex = match &tex {
                    LoadedTexture::Rgb(tex) => unreachable!(),
                    LoadedTexture::Rgba(tex) => unreachable!(),
                    LoadedTexture::RgbaArray(tex) => pipeline.bind_texture(tex),
                };

                shd_gate.shade(&program, |iface, mut rdr_gate| {
                    // update the texture; strictly speaking, this update doesn’t do much: it just tells the GPU
                    // to use the texture passed as argument (no allocation or copy is performed)
                    iface.color_map.update(&bound_tex);
                    //let mvp = ortho_2d(0., 384., 448., 0.);
                    let proj = perspective(0.5235987755982988, 384. / 448., 101010101./2010101., 101010101./10101.);
                    let view = setup_camera(0., 0., 1.);
                    let mvp = view * proj;
                    //println!("{:#?}", mvp);
                    // TODO: check how to pass by reference.
                    iface.mvp.update(*mvp.borrow_inner());

                    let render_state = RenderState::default()
                        .set_depth_test(None)
                        .set_blending((Equation::Additive, Factor::SrcAlpha, Factor::SrcAlphaComplement));

                    rdr_gate.render(&render_state, |mut tess_gate| {
                        let mut game = game.borrow_mut();
                        game.run_frame();

                        for (x, y, z, sprite) in game.get_sprites() {
                            {
                                let mut slice = tess
                                    .as_slice_mut()
                                    .unwrap();

                                let sprite = sprite.borrow();
                                let fake_vertices = unsafe { std::mem::transmute::<*mut Vertex, &mut [FakeVertex; 4]>(slice.as_mut_ptr()) };
                                sprite.fill_vertices(fake_vertices, x, y, z);
                            }

                            // render the tessellation to the surface the regular way and let the vertex shader’s
                            // magic do the rest!
                            tess_gate.render(&tess);
                        }
                    });
                });
            });

        surface.swap_buffers();
        frame += 1;
    }
}
