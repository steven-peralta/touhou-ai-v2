use core::ptr::null;

#[link(name = "glide2x")]
extern "C" {
    fn grGlideInit();
    fn grGlideShutdown();
    fn grSstSelect(sst: u32);
    fn grSstWinOpen(hwnd: u32, resolution: u32, refresh: u32, color_format: u32, origin_location: u32, num_buf: i32, num_aux_buf: i32);
    fn grBufferSwap(interval: i32);
    fn grBufferClear(color: u32, alpha: u8, depth: u16);
    fn grTexCalcMemRequired(min: Lod, max: Lod, aspect: AspectRatio, format: TextureFormat) -> u32;
    fn grTexDownloadMipMap(tmu: u32, start: u32, even_odd: EvenOdd, info: *const TexInfo);
    fn grTexSource(tmu: u32, start: u32, even_odd: EvenOdd, info: &TexInfo);
    fn grTexMinAddress(tmu: u32) -> u32;
    fn grTexMaxAddress(tmu: u32) -> u32;
    fn guAlphaSource(mode: AlphaSource) -> u32;
    fn guColorCombineFunction(function: ColorCombineFnc) -> u32;
    fn grTexCombineFunction(tmu: u32, function: TextureCombineFnc) -> u32;
    fn grDrawTriangle(a: *const Vertex, b: *const Vertex, c: *const Vertex);
    fn grAlphaBlendFunction(a: Blend, b: Blend, c: Blend, d: Blend);
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum Lod {
    L256x256 = 0,
    L128x128 = 1,
    L64x64 = 2,
    L32x32 = 3,
    L16x16 = 4,
    L8x8 = 5,
    L4x4 = 6,
    L2x2 = 7,
    L1x1 = 8,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum AspectRatio {
    A8x1 = 0,
    A4x1 = 1,
    A2x1 = 2,
    A1x1 = 3,
    A1x2 = 4,
    A1x4 = 5,
    A1x8 = 6,
}

fn lod_aspect_from_dimensions(dimensions: (u32, u32)) -> (Lod, AspectRatio) {
    match dimensions {
        (256, 256) => (Lod::L256x256, AspectRatio::A1x1),
        (128, 128) => (Lod::L128x128, AspectRatio::A1x1),
        (64, 64) => (Lod::L64x64, AspectRatio::A1x1),
        (32, 32) => (Lod::L32x32, AspectRatio::A1x1),
        (16, 16) => (Lod::L16x16, AspectRatio::A1x1),
        (8, 8) => (Lod::L8x8, AspectRatio::A1x1),
        (4, 4) => (Lod::L4x4, AspectRatio::A1x1),
        (2, 2) => (Lod::L2x2, AspectRatio::A1x1),
        (1, 1) => (Lod::L1x1, AspectRatio::A1x1),
        (width, height) => todo!("NPOT texture size {width}×{height}"),
    }
}

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum TextureFormat {
    Rgb332 = 0,
    Yiq422 = 1,
    Alpha8 = 2,
    Intensity8 = 3,
    AlphaIntensity44 = 4,
    P8 = 5,
    Argb8332 = 8,
    Ayiq8422 = 9,
    Rgb565 = 10,
    Argb1555 = 11,
    Argb4444 = 12,
    AlphaIntensity88 = 13,
    Ap88 = 14,
}

#[repr(C)]
pub struct TexInfo {
    pub small_lod: Lod,
    pub large_lod: Lod,
    pub aspect: AspectRatio,
    pub format: TextureFormat,
    data: *const u8,
}

impl TexInfo {
    pub fn new(width: u32, height: u32, format: TextureFormat) -> TexInfo {
        let (lod, aspect) = lod_aspect_from_dimensions((width, height));
        TexInfo {
            small_lod: lod,
            large_lod: lod,
            aspect,
            format,
            data: null(),
        }
    }

    pub fn with_data(width: u32, height: u32, format: TextureFormat, data: &[u8]) -> TexInfo {
        let (lod, aspect) = lod_aspect_from_dimensions((width, height));
        TexInfo {
            small_lod: lod,
            large_lod: lod,
            aspect,
            format,
            data: data.as_ptr(),
        }
    }
}

#[repr(C)]
pub struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    r: f32,
    g: f32,
    b: f32,
    ooz: f32,
    a: f32,
    oow: f32,
    sow0: f32,
    tow0: f32,
    oow0: f32,
    sow1: f32,
    tow1: f32,
    oow1: f32,
}

impl Vertex {
    pub fn new(x: f32, y: f32, sow: f32, tow: f32, color: u32) -> Vertex {
        let z = 1.0;
        let r = ((color >> 24) & 0xff) as f32;
        let g = ((color >> 16) & 0xff) as f32;
        let b = ((color >> 8) & 0xff) as f32;
        let a = (color & 0xff) as f32;
        let ooz = 1.0;
        let oow = 1.0;
        let sow0 = sow;
        let tow0 = tow;
        let oow0 = 1.0;
        let sow1 = sow;
        let tow1 = tow;
        let oow1 = 1.0;
        Vertex {
            x, y, z,
            r, g, b,
            ooz,
            a,
            oow,
            sow0, tow0, oow0,
            sow1, tow1, oow1,
        }
    }
}

pub fn glide_init() {
    unsafe { grGlideInit() };
}

pub fn glide_shutdown() {
    unsafe { grGlideShutdown() };
}

pub fn sst_select(sst: u32) {
    unsafe { grSstSelect(sst) };
}

pub fn sst_win_open(width: u32, height: u32, refresh: u32) {
    let resolution = match (width, height) {
        (320, 200) => 0,
        (320, 240) => 1,
        (400, 256) => 2,
        (512, 384) => 3,
        (640, 200) => 4,
        (640, 350) => 5,
        (640, 400) => 6,
        (640, 480) => 7,
        _ => unreachable!("Unknown screen resolution {width}×{height}."),
    };
    let refresh = match refresh {
        60 => 0,
        70 => 1,
        72 => 2,
        75 => 3,
        80 => 4,
        90 => 5,
        100 => 6,
        85 => 7,
        120 => 8,
        _ => unreachable!("Unknown refresh rate {refresh} Hz."),
    };
    let color_format = 2; // RGBA
    let origin_location = 0; // Upper Left
    unsafe { grSstWinOpen(0, resolution, refresh, color_format, origin_location, 2, 0) };
}

pub fn buffer_swap(interval: i32) {
    unsafe { grBufferSwap(interval) };
}

pub fn buffer_clear(color: u32, alpha: u8, depth: u16) {
    unsafe { grBufferClear(color, alpha, depth) };
}

pub fn tex_calc_mem_required(small_lod: Lod, large_lod: Lod, aspect: AspectRatio, format: TextureFormat) -> u32 {
    unsafe { grTexCalcMemRequired(small_lod, large_lod, aspect, format) }
}

pub fn tex_download_mip_map(tmu: u32, start: u32, even_odd: EvenOdd, info: &TexInfo) {
    unsafe { grTexDownloadMipMap(tmu, start, even_odd, info) };
}

pub fn tex_source(tmu: u32, start: u32, even_odd: EvenOdd, info: &TexInfo) {
    unsafe { grTexSource(tmu, start, even_odd, info) };
}

pub fn tex_min_address(tmu: u32) -> u32 {
    unsafe { grTexMinAddress(tmu) }
}

pub fn tex_max_address(tmu: u32) -> u32 {
    unsafe { grTexMaxAddress(tmu) }
}

pub fn alpha_source(mode: AlphaSource) {
    unsafe { guAlphaSource(mode) };
}

pub fn color_combine_function(function: ColorCombineFnc) {
    unsafe { guColorCombineFunction(function) };
}

pub fn tex_combine_function(tmu: u32, function: TextureCombineFnc) {
    unsafe { grTexCombineFunction(tmu, function) };
}

pub fn alpha_blend_function(a: Blend, b: Blend, c: Blend, d: Blend) {
    unsafe { grAlphaBlendFunction(a, b, c, d) };
}

#[repr(i32)]
pub enum EvenOdd {
    Even = 0,
    Odd = 1,
    Both = 2,
}

#[repr(i32)]
pub enum Blend {
    Zero = 0,
    SrcAlpha = 1,
    SrcColor = 2,
    DstAlpha = 3,
    One = 4,
    OneMinusSrcAlpha = 5,
    OneMinusSrcColor = 6,
    OneMinusDstAlpha = 7,
    AlphaSaturate = 15,
}

#[repr(i32)]
pub enum AlphaSource {
    CcAlpha = 0,
    IteratedAlpha = 1,
    TextureAlpha = 2,
    TextureAlphaTimesIteratedAlpha = 3,
}

#[repr(i32)]
pub enum ColorCombineFnc {
    Zero = 0,
    Ccrgb = 1,
    Itrgb = 2,
    ItrgbDelta0 = 3,
    DecalTexture = 4,
    TextureTimesCcrgb = 5,
    TextureTimesItrgb = 6,
    TextureTimesItrgbDelta0 = 7,
    TextureTimesItrgbAddAlpha = 8,
    TextureTimesAlpha = 9,
    TextureTimesAlphaAddItrgb = 10,
    TextureAddItrgb = 11,
    TextureSubItrgb = 12,
    CcrgbBlendItrgbOnTexalpha = 13,
    DiffSpecA = 14,
    DiffSpecB = 15,
    One = 16,
}

#[repr(i32)]
pub enum TextureCombineFnc {
    Zero = 0,
    Decal = 1,
    Other = 2,
    Add = 3,
    Multiply = 4,
    Subtract = 5,
    Detail = 6,
    DetailOther = 7,
    TrilinearOdd = 8,
    TrilinearEven = 9,
    One = 10,
}

pub fn draw_triangle(a: &Vertex, b: &Vertex, c: &Vertex) {
    unsafe { grDrawTriangle(a, b, c) };
}
