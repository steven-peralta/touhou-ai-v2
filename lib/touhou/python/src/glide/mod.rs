use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PySequence};
use pyo3::exceptions::PyTypeError;
use std::collections::{HashMap, BTreeMap};
use image::GenericImageView;

mod gr;

#[inline(always)]
fn pixel_to_rgb332(pixel: [u8; 4]) -> [u8; 1] {
    [(pixel[0] & 0xe0) | ((pixel[1] >> 3) & 0x1c) | (pixel[2] >> 6)]
}

#[inline(always)]
fn pixel_to_argb8332(pixel: [u8; 4]) -> [u8; 2] {
    [(pixel[0] & 0xe0) | ((pixel[1] >> 3) & 0x1c) | (pixel[2] >> 6), pixel[3]]
}

#[inline(always)]
fn pixel_to_argb4444(pixel: [u8; 4]) -> [u8; 2] {
    [(pixel[1] & 0xf0) | (pixel[2] >> 4), (pixel[3] & 0xf0) | (pixel[0] >> 4)]
}

#[inline(always)]
fn pixel_to_argb1555(pixel: [u8; 4]) -> [u8; 2] {
    [((pixel[1] << 2) & 0xe0) | (pixel[2] >> 3), (pixel[3] & 0x80) | ((pixel[0] >> 1) & 0x7c) | (pixel[1] >> 6)]
}

#[inline(always)]
fn pixel_to_rgb565(pixel: [u8; 4]) -> [u8; 2] {
    [((pixel[1] << 3) & 0xe0) | (pixel[2] >> 3), (pixel[0] & 0xf8) | (pixel[1] >> 5)]
}

fn merge_alpha(rgb: &image::DynamicImage, alpha: &image::DynamicImage) -> Vec<u8> {
    let alpha = match alpha.grayscale() {
        image::DynamicImage::ImageLuma8(img) => img,
        foo => panic!("TODO {:?} {:?}", alpha, foo),
    };
    rgb
        .pixels()
        .zip(alpha.pixels())
        .map(|((_x, _y, rgb), alpha)| pixel_to_argb4444([rgb[0], rgb[1], rgb[2], alpha[0]]))
        .flatten()
        .collect::<Vec<_>>()
}

#[derive(Debug)]
struct TextureManager {
    tmu: u32,
    next_tex_location: u32,
    max_tex_location: u32,
    textures: BTreeMap<u32, gr::TextureFormat>,
}

impl TextureManager {
    fn new(tmu: u32) -> TextureManager {
        let next_tex_location = gr::tex_min_address(tmu);
        let max_tex_location = gr::tex_max_address(tmu);
        let textures = BTreeMap::new();
        TextureManager {
            tmu,
            next_tex_location,
            max_tex_location,
            textures,
        }
    }

    fn download(&mut self, tex: &gr::TexInfo) -> PyResult<u32> {
        let location = self.next_tex_location;
        let size = gr::tex_calc_mem_required(tex.small_lod, tex.large_lod, tex.aspect, tex.format);
        if location + size > self.max_tex_location {
            return Err(PyTypeError::new_err("Out of memory"));
        }
        gr::tex_download_mip_map(self.tmu, location, gr::EvenOdd::Both, tex);
        self.next_tex_location += size;
        self.textures.insert(location, tex.format);
        Ok(location)
    }

    fn get(&self, address: u32) -> gr::TexInfo {
        if let Some(&format) = self.textures.get(&address) {
            gr::TexInfo::new(256, 256, format)
        } else {
            unreachable!("Not uploaded texture at address 0x{:08x}!", address);
        }
    }
}

#[pyclass]
struct GameRenderer {
    #[pyo3(get, set)]
    size: (u32, u32, u32, u32),

    texture_manager: TextureManager,
}

#[pymethods]
impl GameRenderer {
    #[new]
    fn new() -> GameRenderer {
        let size = (0, 0, 0, 0);
        let texture_manager = TextureManager::new(0);
        GameRenderer {
            size,
            texture_manager,
        }
    }

    fn start(&self, common: PyObject) {
        gr::color_combine_function(gr::ColorCombineFnc::TextureTimesItrgb);
        gr::alpha_blend_function(gr::Blend::SrcAlpha, gr::Blend::OneMinusSrcAlpha, gr::Blend::One, gr::Blend::Zero);
        gr::alpha_source(gr::AlphaSource::TextureAlphaTimesIteratedAlpha);
        gr::tex_combine_function(0, gr::TextureCombineFnc::Decal);
    }

    fn load_textures(&mut self, py: Python, anms: HashMap<String, Vec<PyObject>>) -> PyResult<()> {
        for (filename, anm) in anms {
            for anm in anm {
                let png_rgb: String = anm.getattr(py, "first_name")?.extract(py)?;
                let png_alpha: Option<String> = anm.getattr(py, "secondary_name")?.extract(py)?;
                let (_, png_rgb) = png_rgb.rsplit_once('/').unwrap();
                use std::path::PathBuf;
                let texture_address = if let Some(png_alpha) = png_alpha {
                    let (_, png_alpha) = png_alpha.rsplit_once('/').unwrap();
                    //image::load_from_memory_with_format(b"", image::ImageFormat::Png).unwrap();
                    let rgb = image::open(["/", "tmp", "touhou", png_rgb].iter().collect::<PathBuf>()).unwrap();
                    let alpha = image::open(["/", "tmp", "touhou", png_alpha].iter().collect::<PathBuf>()).unwrap();
                    assert_eq!(rgb.dimensions(), alpha.dimensions());
                    let (width, height) = rgb.dimensions();
                    let rgba = merge_alpha(&rgb, &alpha);
                    let tex = gr::TexInfo::with_data(width, height, gr::TextureFormat::Argb4444, &rgba);
                    self.texture_manager.download(&tex)?
                } else {
                    //image::load_from_memory_with_format(b"", image::ImageFormat::Png).unwrap();
                    let rgb = image::open(["/", "tmp", "touhou", png_rgb].iter().collect::<PathBuf>()).unwrap();
                    let (width, height) = rgb.dimensions();
                    let rgb = rgb.pixels()
                        .map(|(x, y, rgb)| pixel_to_rgb565([rgb[0], rgb[1], rgb[2], 0xff]))
                        .flatten()
                        .collect::<Vec<_>>();
                    let tex = gr::TexInfo::with_data(width, height, gr::TextureFormat::Rgb565, &rgb);
                    self.texture_manager.download(&tex)?
                };
                anm.setattr(py, "texture", texture_address)?;
                let texture: u32 = anm.getattr(py, "texture")?.extract(py)?;
            }
        }
        Ok(())
    }

    fn load_background(&self, background: PyObject) {
        println!("TODO: GameRenderer::load_background({background})");
    }

    fn render_elements(&self, py: Python, elements: &PyList, shift: (f32, f32)) -> PyResult<()> {
        let module = py.import("pytouhou.ui.glide.sprite")?;
        let get_sprite_rendering_data = module.getattr("get_sprite_rendering_data")?;
        let mut prev_texture = u32::MAX;
        for element in elements.iter() {
            /*
            // TODO: only for enemies.
            let visible: bool = element.getattr("visible")?.extract()?;
            if !visible {
                continue;
            }
            */
            let x: f32 = element.getattr("x")?.extract()?;
            let y: f32 = element.getattr("y")?.extract()?;
            let sprite = element.getattr("sprite")?;
            if !sprite.is_none() {
                let (pos, mut texcoords, color): ([f32; 12], [f32; 4], u32) = get_sprite_rendering_data.call1((sprite,))?.extract()?;
                for coord in &mut texcoords {
                    *coord *= 256.0;
                }
                let anm = sprite.getattr("anm")?;
                let texture = anm.getattr("texture")?.extract()?;
                if texture != prev_texture {
                    let tex = self.texture_manager.get(texture);
                    gr::tex_source(0, texture, gr::EvenOdd::Both, &tex);
                    prev_texture = texture;
                }
                draw_triangle(x + shift.0, y + shift.1, pos, texcoords, color);
            }
        }
        Ok(())
    }

    fn render(&self, py: Python, game: PyObject) -> PyResult<()> {
        gr::buffer_clear(0x000000ff, 0xff, 0xffff);
        for things in ["enemies", "effects", "players_bullets"/*, "lasers_sprites()"*/, "players"/*, "msg_sprites()"*/, "bullets", "lasers", "cancelled_bullets", "items", "labels"] {
            let things = game.getattr(py, things)?;
            let things: &PyList = things.extract(py)?;
            self.render_elements(py, things, (32.0, 16.0))?;
        }
        let interface = game.getattr(py, "interface")?;
        let boss = game.getattr(py, "boss")?;
        self.render_interface(py, interface, !boss.is_none(py))?;
        Ok(())
    }

    fn render_interface(&self, py: Python, interface: PyObject, boss: bool) -> PyResult<()> {
        let items = interface.getattr(py, "items")?;
        let items: &PyList = items.extract(py)?;
        self.render_elements(py, items, (0.0, 0.0))?;
        /*
        // TODO: figure out why this doesn’t render alphanumeric characters.
        let labels = interface.getattr(py, "labels")?;
        let labels: &PyDict = labels.extract(py)?;
        self.render_elements(py, labels.values(), (0.0, 0.0))?;
        */
        if boss {
            let items = interface.getattr(py, "boss_items")?;
            let items: &PyList = items.extract(py)?;
            self.render_elements(py, items, (0.0, 0.0))?;
        }
        Ok(())
    }
}

fn draw_triangle(ox: f32, oy: f32, pos: [f32; 12], texcoords: [f32; 4], color: u32) {
    let a = gr::Vertex::new(ox + pos[0], oy + pos[4], texcoords[0], texcoords[2], color);
    let b = gr::Vertex::new(ox + pos[1], oy + pos[5], texcoords[1], texcoords[2], color);
    let c = gr::Vertex::new(ox + pos[2], oy + pos[6], texcoords[1], texcoords[3], color);
    let d = gr::Vertex::new(ox + pos[3], oy + pos[7], texcoords[0], texcoords[3], color);
    gr::draw_triangle(&a, &b, &c);
    gr::draw_triangle(&a, &c, &d);
}

#[pyfunction]
fn init(options: HashMap<String, String>) {
    gr::glide_init();
    gr::sst_select(0);
}

#[pyfunction]
fn shutdown() {
    gr::glide_shutdown();
}

#[pyfunction]
fn create_window(title: &str, posx: u32, posy: u32, width: u32, height: u32, frameskip: u32) {
    gr::sst_win_open(640, 480, 60);
}

#[pyfunction]
fn buffer_swap() {
    gr::buffer_swap(1);
}

pub fn module(py: Python) -> PyResult<&PyModule> {
    let m = PyModule::new(py, "glide")?;
    m.add_class::<GameRenderer>()?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    m.add_function(wrap_pyfunction!(create_window, m)?)?;
    m.add_function(wrap_pyfunction!(buffer_swap, m)?)?;
    Ok(&m)
}
