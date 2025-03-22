use luminance::blending::{Equation, Factor};
use luminance::context::GraphicsContext;
use luminance::pipeline::{BoundTexture, PipelineState};
use luminance::pixel::NormUnsigned;
use luminance::render_state::RenderState;
use luminance::shader::program::{Program, Uniform};
use luminance::tess::{Mode, TessBuilder};
use luminance::texture::Dim2;
use luminance_derive::{Semantics, Vertex, UniformInterface};
use luminance_glfw::{Action, Key, WindowEvent, GlfwSurface, Surface, WindowDim, WindowOpt};
use touhou_formats::th06::anm0::Anm0;
use touhou_formats::th06::ecl::{Ecl, Rank};
use touhou_interpreters::th06::anm0::{Sprite, Vertex as FakeVertex};
use touhou_interpreters::th06::ecl::EclRunner;
use touhou_interpreters::th06::enemy::{Enemy, Game, Position};
use touhou_utils::math::{perspective, setup_camera};
use touhou_utils::prng::Prng;
use std::cell::RefCell;
use std::rc::Rc;
use std::env;
use std::path::Path;

use touhou_runners::common::{load_file_into_vec, load_anm_image, LoadedTexture};

const VS: &str = r#"
in ivec3 in_position;
in vec2 in_texcoord;
in uvec4 in_color;

uniform mat4 mvp;

out vec2 texcoord;
out vec4 color;

void main()
{
    gl_Position = mvp * vec4(vec3(in_position), 1.0);
    texcoord = vec2(in_texcoord);

    // Normalized from the u8 being passed.
    color = vec4(in_color) / 255.;
}
"#;

const FS: &str = r#"
in vec2 texcoord;
in vec4 color;

uniform sampler2D color_map;

out vec4 frag_color;

void main()
{
    frag_color = texture(color_map, texcoord) * color;
}
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Semantics)]
pub enum Semantics {
    #[sem(name = "in_position", repr = "[i16; 3]", wrapper = "VertexPosition")]
    Position,

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
    uv: VertexTexcoord,
    rgba: VertexColor,
}

#[derive(UniformInterface)]
struct ShaderInterface {
    // the 'static lifetime acts as “anything” here
    color_map: Uniform<&'static BoundTexture<'static, Dim2, NormUnsigned>>,

    #[uniform(name = "mvp")]
    mvp: Uniform<[[f32; 4]; 4]>,
}

fn main() {
    // Parse arguments.
    let args: Vec<_> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <ECL file> <ANM file> <easy|normal|hard|lunatic> <sub number>", args[0]);
        return;
    }
    let ecl_filename = Path::new(&args[1]);
    let anm_filename = Path::new(&args[2]);
    let rank: Rank = args[3].parse().expect("rank");
    let sub: u16 = args[4].parse().expect("number");

    // Open the ECL file.
    let buf = load_file_into_vec(ecl_filename).unwrap();
    let (_, ecl) = Ecl::from_slice(&buf).unwrap();

    // Open the ANM file.
    let buf = load_file_into_vec(anm_filename).unwrap();
    let (_, mut anms) = Anm0::from_slice(&buf).unwrap();
    let anm0 = anms.pop().unwrap();
    let anm0 = Rc::new(RefCell::new([anm0.clone(), anm0]));

    if ecl.subs.len() < sub as usize {
        eprintln!("This ecl doesn’t contain a sub named {}.", sub);
        return;
    }

    // Get the time since January 1970 as a seed for the PRNG.
    let time = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let prng = Rc::new(RefCell::new(Prng::new(time.subsec_micros() as u16)));

    // Create the Game god object.
    let game = Game::new(prng, rank);
    let game = Rc::new(RefCell::new(game));

    // And the enemy object.
    let enemy = Enemy::new(Position::new(0., 0.), 500, 0, 640, false, Rc::downgrade(&anm0), Rc::downgrade(&game));
    let mut ecl_runner = EclRunner::new(&ecl, enemy.clone(), sub);

    assert_eq!(std::mem::size_of::<Vertex>(), std::mem::size_of::<FakeVertex>());
    let vertices: [Vertex; 4] = {
        let data = std::mem::MaybeUninit::uninit();
        unsafe { data.assume_init() }
    };

    let mut surface = GlfwSurface::new(WindowDim::Windowed(384, 448), "Touhou", WindowOpt::default()).unwrap();

    // Open the image atlas matching this ANM.
    let tex = load_anm_image(&mut surface, &anm0.borrow()[0], &anm_filename).expect("image loading");

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

        if ecl_runner.running == false {
            break;
        }

        {
            let mut slice = tess
                .as_slice_mut()
                .unwrap();

            ecl_runner.run_frame();
            {
                let mut enemy = enemy.borrow_mut();
                enemy.update();
            }
            let mut game = game.borrow_mut();
            game.run_frame();
            let sprites = game.get_sprites();
            fill_vertices_ptr(sprites, slice.as_mut_ptr());
        }

        // here, we need to bind the pipeline variable; it will enable us to bind the texture to the GPU
        // and use it in the shader
        surface
            .pipeline_builder()
            .pipeline(&back_buffer, &PipelineState::default(), |pipeline, mut shd_gate| {
                // bind our fancy texture to the GPU: it gives us a bound texture we can use with the shader
                let bound_tex = match &tex {
                    LoadedTexture::Rgb(tex) => pipeline.bind_texture(tex),
                    LoadedTexture::Rgba(tex) => pipeline.bind_texture(tex),
                    LoadedTexture::RgbaArray(tex) => unreachable!(),
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
                        .set_blending((Equation::Additive, Factor::SrcAlpha, Factor::SrcAlphaComplement));

                    rdr_gate.render(&render_state, |mut tess_gate| {
                        // render the tessellation to the surface the regular way and let the vertex shader’s
                        // magic do the rest!
                        tess_gate.render(&tess);
                    });
                });
            });

        surface.swap_buffers();
    }
}

fn fill_vertices_ptr(sprites: Vec<(f32, f32, f32, Rc<RefCell<Sprite>>)>, vertices: *mut Vertex) {
    let mut fake_vertices = unsafe { std::mem::transmute::<*mut Vertex, &mut [FakeVertex; 4]>(vertices) };
    for (x, y, z, sprite) in sprites {
        let sprite = sprite.borrow();
        sprite.fill_vertices(&mut fake_vertices, x, y, z);
    }
}
