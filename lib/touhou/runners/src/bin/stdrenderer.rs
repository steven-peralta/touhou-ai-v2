use luminance::blending::{Equation, Factor};
use luminance::context::GraphicsContext;
use luminance::pipeline::{BoundTexture, PipelineState};
use luminance::pixel::NormUnsigned;
use luminance::render_state::RenderState;
use luminance::shader::program::{Program, Uniform};
use luminance::tess::{Mode, TessBuilder, TessSliceIndex};
use luminance::texture::Dim2;
use luminance_derive::{Semantics, Vertex, UniformInterface};
use luminance_glfw::{Action, Key, WindowEvent, GlfwSurface, Surface, WindowDim, WindowOpt};
use touhou_formats::th06::anm0::Anm0;
use touhou_formats::th06::std::{Stage, Position, Box2D};
use touhou_interpreters::th06::anm0::{AnmRunner, Sprite, Vertex as FakeVertex};
use touhou_interpreters::th06::std::StageRunner;
use touhou_utils::prng::Prng;
use touhou_utils::math::perspective;
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
uniform vec3 instance_position;

out vec2 texcoord;
out vec4 color;

void main()
{
    vec3 position = vec3(in_position) + instance_position;
    gl_Position = mvp * vec4(position, 1.0);
    texcoord = vec2(in_texcoord);

    // Normalized from the u8 being passed.
    color = vec4(in_color) / 255.;
}
"#;

const FS: &str = r#"
in vec2 texcoord;
in vec4 color;

uniform sampler2D color_map;
uniform float fog_scale;
uniform float fog_end;
uniform vec4 fog_color;

out vec4 frag_color;

void main()
{
    vec4 temp_color = texture(color_map, texcoord) * color;
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float fog_density = clamp((fog_end - depth) * fog_scale, 0.0, 1.0);
    frag_color = vec4(mix(fog_color, temp_color, fog_density).rgb, temp_color.a);
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

    #[uniform(name = "instance_position")]
    instance_position: Uniform<[f32; 3]>,

    #[uniform(name = "fog_scale")]
    fog_scale: Uniform<f32>,

    #[uniform(name = "fog_end")]
    fog_end: Uniform<f32>,

    #[uniform(name = "fog_color")]
    fog_color: Uniform<[f32; 4]>,
}

fn main() {
    // Parse arguments.
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <STD file> <ANM file>", args[0]);
        return;
    }
    let std_filename = Path::new(&args[1]);
    let anm_filename = Path::new(&args[2]);

    // Open the STD file.
    let buf = load_file_into_vec(std_filename).unwrap();
    let (_, stage) = Stage::from_slice(&buf).unwrap();

    // Open the ANM file.
    let buf = load_file_into_vec(anm_filename).unwrap();
    let (_, mut anms) = Anm0::from_slice(&buf).unwrap();
    let anm0 = anms.pop().unwrap();

    // TODO: seed this PRNG with a valid seed.
    let prng = Rc::new(RefCell::new(Prng::new(0)));

    let mut surface = GlfwSurface::new(WindowDim::Windowed(384, 448), "Touhou", WindowOpt::default()).unwrap();

    // Open the image atlas matching this ANM.
    let tex = load_anm_image(&mut surface, &anm0, anm_filename).expect("image loading");

    assert_eq!(std::mem::size_of::<Vertex>(), std::mem::size_of::<FakeVertex>());
    let mut vertices: Vec<Vertex> = vec![];
    let mut indices = vec![];

    {
        let anms = Rc::new(RefCell::new([anm0]));
        for model in stage.models.iter() {
            let begin = vertices.len();
            for quad in model.quads.iter() {
                let Position { x, y, z } = quad.pos;
                let Box2D { width, height } = quad.size_override;

                // Create the AnmRunner from the ANM and the sprite.
                let sprite = Rc::new(RefCell::new(Sprite::with_size(width, height)));
                let _anm_runner = AnmRunner::new(anms.clone(), quad.anm_script as u8, sprite.clone(), Rc::downgrade(&prng), 0);
                let mut new_vertices: [Vertex; 6] = {
                    let data = std::mem::MaybeUninit::uninit();
                    unsafe { data.assume_init() }
                };
                fill_vertices(sprite.clone(), &mut new_vertices, x, y, z);
                new_vertices[4] = new_vertices[0];
                new_vertices[5] = new_vertices[2];
                vertices.extend(&new_vertices);
            }
            let end = vertices.len();
            indices.push((begin, end));
        }
    }

    let mut stage_runner = StageRunner::new(Rc::new(RefCell::new(stage)));

    // set the uniform interface to our type so that we can read textures from the shader
    let program =
        Program::<Semantics, (), ShaderInterface>::from_strings(None, VS, None, FS).expect("program creation").ignore_warnings();

    let tess = TessBuilder::new(&mut surface)
        .add_vertices(vertices)
        .set_mode(Mode::Triangle)
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

        {
            stage_runner.run_frame();
            //let sprites = stage.get_sprites();
            //fill_vertices_ptr(sprites, slice.as_mut_ptr());
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

                    let proj = perspective(0.5235987755982988, 384. / 448., 101010101./2010101., 101010101./10101.);
                    let model_view = stage_runner.get_model_view();
                    let mvp = model_view * proj;
                    // TODO: check how to pass by reference.
                    iface.mvp.update(*mvp.borrow_inner());

                    let near = stage_runner.fog_near - 101010101. / 2010101.;
                    let far = stage_runner.fog_far - 101010101. / 2010101.;
                    iface.fog_color.update(stage_runner.fog_color);
                    iface.fog_scale.update(1. / (far - near));
                    iface.fog_end.update(far);

                    let render_state = RenderState::default()
                        .set_blending((Equation::Additive, Factor::SrcAlpha, Factor::SrcAlphaComplement));

                    let stage = stage_runner.stage.borrow();
                    for instance in stage.instances.iter() {
                        iface.instance_position.update([instance.pos.x, instance.pos.y, instance.pos.z]);

                        rdr_gate.render(&render_state, |mut tess_gate| {
                            let (begin, end) = indices[instance.id as usize];
                            tess_gate.render(tess.slice(begin..end));
                        });
                    }
                });
            });

        surface.swap_buffers();
    }
}

fn fill_vertices(sprite: Rc<RefCell<Sprite>>, vertices: &mut [Vertex; 6], x: f32, y: f32, z: f32) {
    let mut fake_vertices = unsafe { std::mem::transmute::<&mut [Vertex; 6], &mut [FakeVertex; 4]>(vertices) };
    let sprite = sprite.borrow();
    sprite.fill_vertices(&mut fake_vertices, x, y, z);
}
