use ears::{Music, AudioController};
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
use touhou_formats::th06::pbg3;
use touhou_formats::th06::anm0::Anm0;
use touhou_interpreters::th06::anm0::{AnmRunner, Sprite, Vertex as FakeVertex};
use touhou_utils::math::{perspective, setup_camera, ortho_2d};
use touhou_utils::prng::Prng;
use std::cell::RefCell;
use std::rc::Rc;
use std::env;
use std::path::Path;

use touhou_runners::common::{self, LoadedTexture};

const VS: &str = r#"
in ivec3 in_position;
in vec2 in_texcoord;
in vec4 in_color;

uniform mat4 mvp;

out vec2 texcoord;
out vec4 color;

void main()
{
    gl_Position = mvp * vec4(vec3(in_position), 1.0);
    texcoord = vec2(in_texcoord);

    // It’s already normalized from the u8 being passed.
    color = in_color;
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
    #[vertex(normalized = "true")]
    rgba: VertexColor,
}

#[derive(UniformInterface)]
struct ShaderInterface {
    // the 'static lifetime acts as “anything” here
    color_map: Uniform<&'static BoundTexture<'static, Dim2, NormUnsigned>>,

    #[uniform(name = "mvp")]
    mvp: Uniform<[[f32; 4]; 4]>,
}

const DEFAULT_VERTICES: [Vertex; 4] = [
    Vertex::new(VertexPosition::new([0, 0, 0]), VertexTexcoord::new([0., 0.]), VertexColor::new([255, 255, 255, 255])),
    Vertex::new(VertexPosition::new([640, 0, 0]), VertexTexcoord::new([1., 0.]), VertexColor::new([255, 255, 255, 255])),
    Vertex::new(VertexPosition::new([640, 480, 0]), VertexTexcoord::new([1., 1.]), VertexColor::new([255, 255, 255, 255])),
    Vertex::new(VertexPosition::new([0, 480, 0]), VertexTexcoord::new([0., 1.]), VertexColor::new([255, 255, 255, 255])),
];

fn main() {
    // Parse arguments.
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <unarchived directory>", args[0]);
        return;
    }
    let directory = Path::new(&args[1]);

    let in_dat = directory.join("IN.DAT");
    // Since GLFW can be slow to create its window, let’s decode the splash screen in another
    // thread in the meantime.
    let jpeg_thread = std::thread::spawn(|| {
        let mut in_pbg3 = pbg3::from_path_buffered(in_dat).expect("IN.DAT present");
        let jpeg = in_pbg3.get_file("th06logo.jpg", true).expect("th06logo.jpg in IN.DAT");
        let image = common::load_from_data(&jpeg).expect("th06logo.jpg decodable");
        image
    });

    let music_filename = directory.join("bgm").join("th06_01.wav");
    let music_filename = music_filename.to_str().expect("non-UTF-8 music filename");
    let music = match Music::new(music_filename) {
        Ok(mut music) => {
            music.set_looping(true);
            music.play();
            music
        }
        Err(err) => {
            eprintln!("Impossible to open or play music file: {}", err);
            return;
        }
    };

    let mut surface = GlfwSurface::new(WindowDim::Windowed(640, 480), "Touhou", WindowOpt::default()).expect("GLFW window");

    let image = jpeg_thread.join().expect("image loading");
    let background = common::upload_texture_from_rgb_image(&mut surface, image).expect("upload data to texture");

    let mut background = match background {
        LoadedTexture::Rgb(tex) => tex,
        LoadedTexture::Rgba(tex) => unreachable!(),
        LoadedTexture::RgbaArray(tex) => unreachable!(),
    };

    // set the uniform interface to our type so that we can read textures from the shader
    let program =
        Program::<Semantics, (), ShaderInterface>::from_strings(None, VS, None, FS).expect("program creation").ignore_warnings();

    let mut tess = TessBuilder::new(&mut surface)
        .add_vertices(DEFAULT_VERTICES)
        .set_mode(Mode::TriangleFan)
        .build()
        .unwrap();

    let tl_dat = directory.join("TL.DAT");
    let mut tl_pbg3 = pbg3::from_path_buffered(tl_dat).expect("TL.DAT present");

    let mut back_buffer = surface.back_buffer().unwrap();
    let mut resize = false;
    let mut frame = 0;
    let mut z_pressed = false;
    let mut x_pressed = false;

    'app: loop {
        for event in surface.poll_events() {
            match event {
                WindowEvent::Close | WindowEvent::Key(Key::Escape, _, Action::Release, _) => break 'app,

                WindowEvent::Key(Key::Z, _, Action::Press, _) => z_pressed = true,
                WindowEvent::Key(Key::X, _, Action::Press, _) => x_pressed = true,

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

        frame += 1;
        if frame == 60 {
            let jpeg = tl_pbg3.get_file("title00.jpg", true).expect("title00.jpg in TL.DAT");
            let image = common::load_from_data(&jpeg).expect("th06logo.jpg decodable");
            common::reupload_texture_from_rgb_image(&mut background, image).expect("upload data to texture");
        }

        if frame >= 60 && z_pressed {
            let jpeg = tl_pbg3.get_file("select00.jpg", true).expect("select00.jpg in TL.DAT");
            let image = common::load_from_data(&jpeg).expect("select00.jpg decodable");
            common::reupload_texture_from_rgb_image(&mut background, image).expect("upload data to texture");
        }

        // here, we need to bind the pipeline variable; it will enable us to bind the texture to the GPU
        // and use it in the shader
        surface
            .pipeline_builder()
            .pipeline(&back_buffer, &PipelineState::default(), |pipeline, mut shd_gate| {
                // bind our fancy texture to the GPU: it gives us a bound texture we can use with the shader
                let tex = pipeline.bind_texture(&background);

                shd_gate.shade(&program, |iface, mut rdr_gate| {
                    // update the texture; strictly speaking, this update doesn’t do much: it just tells the GPU
                    // to use the texture passed as argument (no allocation or copy is performed)
                    iface.color_map.update(&tex);
                    let mvp = ortho_2d(0., 640., 480., 0.);
                    // TODO: check how to pass by reference.
                    iface.mvp.update(*mvp.borrow_inner());

                    let render_state = RenderState::default()
                        .set_blending((Equation::Additive, Factor::SrcAlpha, Factor::SrcAlphaComplement));

                    rdr_gate.render(&render_state, |mut tess_gate| {
                        tess_gate.render(&tess);
                    });
                });
            });

        surface.swap_buffers();
    }
}
