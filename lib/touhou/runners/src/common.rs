use image::{GenericImageView, DynamicImage, GrayImage, ImageError};
use luminance::pixel::{NormRGB8UI, NormRGBA8UI};
use luminance::texture::{Dim2, Dim2Array, Sampler, Texture, GenMipmaps};
use luminance_glfw::GlfwSurface;
use touhou_formats::th06::anm0::Anm0;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub fn load_file_into_vec<P: AsRef<Path>>(filename: P) -> io::Result<Vec<u8>> {
    let file = File::open(filename)?;
    let mut file = BufReader::new(file);
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

pub enum LoadedTexture {
    Rgba(Texture<Dim2, NormRGBA8UI>),
    Rgb(Texture<Dim2, NormRGB8UI>),
    RgbaArray(Texture<Dim2Array, NormRGBA8UI>),
}

#[derive(Debug)]
pub enum TextureLoadError {
    CannotOpenRgb(String, ImageError),
    CannotOpenAlpha(String, ImageError),
    AlphaToGrayscale(String),
}

fn open_rgb_png(path: &Path) -> Result<DynamicImage, TextureLoadError> {
    // load the texture into memory as a whole bloc (i.e. no streaming)
    image::open(&path).map_err(|e| TextureLoadError::CannotOpenRgb(path.to_str().unwrap().to_owned(), e))
}

fn open_alpha_png(path: &Path) -> Result<DynamicImage, TextureLoadError> {
    // load the texture into memory as a whole bloc (i.e. no streaming)
    image::open(&path).map_err(|e| TextureLoadError::CannotOpenAlpha(path.to_str().unwrap().to_owned(), e))
}

fn merge_rgb_alpha(rgb: &DynamicImage, alpha: &GrayImage) -> Vec<(u8, u8, u8, u8)> {
    rgb
        .pixels()
        .zip(alpha.pixels())
        .map(|((_x, _y, rgb), luma)| (rgb[0], rgb[1], rgb[2], luma[0]))
        .collect::<Vec<_>>()
}

pub fn load_from_data(data: &[u8]) -> Result<DynamicImage, ImageError> {
    image::load_from_memory(data)
}

pub fn reupload_texture_from_rgb_image(tex: &mut Texture<Dim2, NormRGB8UI>, img: DynamicImage) -> Result<(), TextureLoadError> {
    let texels = img
        .pixels()
        .map(|(_x, _y, rgb)| (rgb[0], rgb[1], rgb[2]))
        .collect::<Vec<_>>();

    // the first argument disables mipmap generation (we don’t care so far)
    tex.upload(GenMipmaps::No, &texels).unwrap();

    Ok(())
}

pub fn upload_texture_from_rgb_image(surface: &mut GlfwSurface, img: DynamicImage) -> Result<LoadedTexture, TextureLoadError> {
    let (width, height) = img.dimensions();

    // create the luminance texture; the third argument is the number of mipmaps we want (leave it
    // to 0 for now) and the latest is a the sampler to use when sampling the texels in the
    // shader (we’ll just use the default one)
    let mut tex =
        Texture::new(surface, [width, height], 0, Sampler::default()).expect("luminance texture creation");

    reupload_texture_from_rgb_image(&mut tex, img)?;

    Ok(LoadedTexture::Rgb(tex))
}

pub fn load_rgb_texture(surface: &mut GlfwSurface, path: &Path) -> Result<LoadedTexture, TextureLoadError> {
    let img = open_rgb_png(&path)?;
    upload_texture_from_rgb_image(surface, img)
}

fn load_rgb_a_pngs(surface: &mut GlfwSurface, rgb: &Path, alpha: &Path) -> Result<LoadedTexture, TextureLoadError> {
    let img = open_alpha_png(&alpha)?;
    let alpha = match img.grayscale() {
        DynamicImage::ImageLuma8(img) => img,
        _ => {
            return Err(TextureLoadError::AlphaToGrayscale(alpha.to_str().unwrap().to_owned()))
        }
    };
    let (width, height) = img.dimensions();
    let img = open_rgb_png(&rgb)?;
    assert_eq!((width, height), img.dimensions());
    let texels = merge_rgb_alpha(&img, &alpha);

    // create the luminance texture; the third argument is the number of mipmaps we want (leave it
    // to 0 for now) and the latest is a the sampler to use when sampling the texels in the
    // shader (we’ll just use the default one)
    let tex =
        Texture::new(surface, [width, height], 0, Sampler::default()).expect("luminance texture creation");

    // the first argument disables mipmap generation (we don’t care so far)
    tex.upload(GenMipmaps::No, &texels).unwrap();

    Ok(LoadedTexture::Rgba(tex))
}

pub fn load_anm_image<P: AsRef<Path>>(mut surface: &mut GlfwSurface, anm0: &Anm0, anm_filename: P) -> Result<LoadedTexture, TextureLoadError> {
    let anm_filename = anm_filename.as_ref();
    let png_filename = anm_filename.with_file_name(Path::new(&anm0.png_filename).file_name().unwrap());
    match anm0.alpha_filename {
        Some(ref filename) => {
            let alpha_filename = anm_filename.with_file_name(Path::new(filename).file_name().unwrap());
            load_rgb_a_pngs(&mut surface, &png_filename, &alpha_filename)
        },
        None => {
            load_rgb_texture(&mut surface, &png_filename)
        }
    }
}

fn load_array_texture(surface: &mut GlfwSurface, images: &[(&Path, &Path)]) -> Result<LoadedTexture, TextureLoadError> {
    let mut decoded = vec![];
    let dimensions = (256, 256);
    for (rgb, alpha) in images {
        let img = open_alpha_png(&alpha)?;
        assert_eq!(dimensions, img.dimensions());
        let alpha = match img.grayscale() {
            DynamicImage::ImageLuma8(img) => img,
            _ => {
                return Err(TextureLoadError::AlphaToGrayscale(alpha.to_str().unwrap().to_owned()))
            }
        };
        let img = open_rgb_png(&rgb)?;
        assert_eq!(dimensions, img.dimensions());
        let texels = merge_rgb_alpha(&img, &alpha);
        decoded.push(texels);
    }

    // create the luminance texture; the third argument is the number of mipmaps we want (leave it
    // to 0 for now) and the latest is a the sampler to use when sampling the texels in the
    // shader (we’ll just use the default one)
    let tex =
        Texture::new(surface, ([dimensions.0, dimensions.1], images.len() as u32), 0, Sampler::default()).expect("luminance texture creation");

    // the first argument disables mipmap generation (we don’t care so far)
    tex.upload(GenMipmaps::No, &decoded.into_iter().flatten().collect::<Vec<_>>()).unwrap();

    Ok(LoadedTexture::RgbaArray(tex))
}

pub fn load_multiple_anm_images<P: AsRef<Path>>(mut surface: &mut GlfwSurface, anms: &[Anm0], anm_filename: P) -> Result<LoadedTexture, TextureLoadError> {
    let anm_filename = anm_filename.as_ref();
    let mut paths = vec![];
    for anm0 in anms.iter() {
        let rgb_filename = anm_filename.with_file_name(Path::new(&anm0.png_filename).file_name().unwrap());
        let filename = anm0.alpha_filename.as_ref().expect("Can’t not have alpha here!");
        let alpha_filename = anm_filename.with_file_name(Path::new(filename).file_name().unwrap());
        paths.push((rgb_filename, alpha_filename));
    }
    let paths: Vec<_> = paths.iter().map(|(rgb, alpha)| (rgb.as_ref(), alpha.as_ref())).collect();
    load_array_texture(&mut surface, paths.as_slice())
}
