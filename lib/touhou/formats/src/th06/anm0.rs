//! ANM0 animation format support.

use nom::{
    IResult,
    bytes::complete::{tag, take_while_m_n},
    number::complete::{le_u8, le_u16, le_u32, le_i32, le_f32},
    sequence::tuple,
    multi::{many_m_n, many0},
};
use std::collections::BTreeMap;

/// Coordinates of a sprite into the image.
#[derive(Debug, Clone)]
pub struct Sprite {
    /// Index inside the anm0.
    pub index: u32,

    /// X coordinate in the sprite sheet.
    pub x: f32,

    /// Y coordinate in the sprite sheet.
    pub y: f32,

    /// Width of the sprite.
    pub width: f32,

    /// Height of the sprite.
    pub height: f32,
}

/// A single instruction, part of a `Script`.
#[derive(Debug, Clone)]
pub struct Call {
    /// Time at which this instruction will be called.
    pub time: u16,

    /// The instruction to call.
    pub instr: Instruction,
}

/// Script driving an animation.
#[derive(Debug, Clone)]
pub struct Script {
    /// List of instructions in this script.
    pub instructions: Vec<Call>,

    /// List of interrupts in this script.
    pub interrupts: BTreeMap<i32, u8>
}

/// Main struct of the ANM0 animation format.
#[derive(Debug, Clone)]
pub struct Anm0 {
    /// Resolution of the image used by this ANM.
    pub size: (u32, u32),

    /// Format of this ANM.
    // TODO: use an enum for that.
    pub format: u32,

    /// Color key, probably used for transparency.
    pub color_key: u32,

    /// File name of the main image.
    pub png_filename: String,

    /// File name of an alpha channel image.
    pub alpha_filename: Option<String>,

    /// A list of sprites, coordinates into the attached image.
    pub sprites: Vec<Sprite>,

    /// A map of scripts.
    pub scripts: BTreeMap<u8, Script>,
}

impl Anm0 {
    /// Parse a slice of bytes into an `Anm0` struct.
    pub fn from_slice(data: &[u8]) -> IResult<&[u8], Vec<Anm0>> {
        many0(parse_anm0)(data)
    }

    /// TODO
    pub fn inv_size(&self) -> (f32, f32) {
        let (x, y) = self.size;
        (1. / x as f32, 1. / y as f32)
    }
}

fn parse_name(i: &[u8]) -> IResult<&[u8], String> {
    let (_, slice) = take_while_m_n(0, 32, |c| c != 0)(i)?;
    let string = match String::from_utf8(slice.to_vec()) {
        Ok(string) => string,
        // XXX: use a more specific error instead.
        Err(_) => return Err(nom::Err::Failure(nom::error::Error::new(i, nom::error::ErrorKind::Eof)))
    };
    Ok((i, string))
}

fn parse_sprite(i: &[u8]) -> IResult<&[u8], Sprite> {
    let (i, (index, x, y, width, height)) = tuple((le_u32, le_f32, le_f32, le_f32, le_f32))(i)?;
    Ok((i, Sprite {
        index,
        x,
        y,
        width,
        height,
    }))
}

macro_rules! declare_anm_instructions {
    ($($opcode:tt => fn $name:ident($($arg:ident: $arg_type:ident),*)),*,) => {
        /// Available instructions in an `Anm0`.
        #[allow(missing_docs)]
        #[derive(Debug, Clone, Copy)]
        pub enum Instruction {
            $(
                $name($($arg_type),*)
            ),*
        }

        fn parse_instruction_args(mut i: &[u8], opcode: u8) -> IResult<&[u8], Instruction> {
            let instr = match opcode {
                $(
                    $opcode => {
                        $(
                            let (i2, $arg) = concat_idents!(le_, $arg_type)(i)?;
                            i = i2;
                        )*
                        Instruction::$name($($arg),*)
                    }
                )*
                // XXX: use a more specific error instead.
                _ => return Err(nom::Err::Failure(nom::error::Error::new(i, nom::error::ErrorKind::Eof)))
            };
            Ok((i, instr))
        }
    };
}

declare_anm_instructions!{
    0 => fn Delete(),
    1 => fn LoadSprite(sprite_number: u32),
    2 => fn SetScale(sx: f32, sy: f32),
    3 => fn SetAlpha(alpha: u32),
    4 => fn SetColor(red: u8, green: u8, blue: u8/*, XXX: x8*/),
    5 => fn Jump(instruction: u32),
    7 => fn ToggleMirrored(),
    9 => fn SetRotations3d(x: f32, y: f32, z: f32),
    10 => fn SetRotationsSpeed3d(x: f32, y: f32, z: f32),
    11 => fn SetScaleSpeed(sx: f32, sy: f32),
    12 => fn Fade(alpha: u32, duration: u32),
    13 => fn SetBlendmodeAdd(),
    14 => fn SetBlendmodeAlphablend(),
    15 => fn KeepStill(),
    16 => fn LoadRandomSprite(min_index: u32, amplitude: u32),
    17 => fn Move(x: f32, y: f32, z: f32),
    18 => fn MoveToLinear(x: f32, y: f32, z: f32, duration: u32),
    19 => fn MoveToDecel(x: f32, y: f32, z: f32, duration: u32),
    20 => fn MoveToAccel(x: f32, y: f32, z: f32, duration: u32),
    21 => fn Wait(),
    22 => fn InterruptLabel(label: i32),
    23 => fn SetCornerRelativePlacement(),
    24 => fn WaitEx(),
    25 => fn SetAllowOffset(allow: u32), // TODO: better name
    26 => fn SetAutomaticOrientation(automatic: u32),
    27 => fn ShiftTextureX(dx: f32),
    28 => fn ShiftTextureY(dy: f32),
    29 => fn SetVisible(visible: u32),
    30 => fn ScaleIn(sx: f32, sy: f32, duration: u32),
    31 => fn Todo(todo: u32),
}

fn parse_anm0(input: &[u8]) -> IResult<&[u8], Anm0> {
    let (i, (num_sprites, num_scripts, _, width, height, format, color_key,
             first_name_offset, _, second_name_offset, version, _,
             _texture_offset, has_data, _next_offset, _)) =
        tuple((le_u32, le_u32, tag(b"\0\0\0\0"), le_u32, le_u32, le_u32, le_u32, le_u32,
               tag(b"\0\0\0\0"), le_u32, le_u32, tag(b"\0\0\0\0"), le_u32, le_u32, le_u32,
               tag(b"\0\0\0\0")))(input)?;

    assert_eq!(version, 0);
    assert_eq!(has_data, 0);
    let num_sprites = num_sprites as usize;
    let num_scripts = num_scripts as usize;

    let (i, sprite_offsets) = many_m_n(num_sprites, num_sprites, le_u32)(i)?;
    let (_, script_offsets) = many_m_n(num_scripts, num_scripts, tuple((le_u32, le_u32)))(i)?;

    let png_filename = if first_name_offset > 0 {
        if input.len() < first_name_offset as usize {
            return Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
        }
        let i = &input[first_name_offset as usize..];
        let (_, name) = parse_name(i)?;
        name
    } else {
        String::new()
    };

    let alpha_filename = if second_name_offset > 0 {
        if input.len() < second_name_offset as usize {
            return Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
        }
        let i = &input[second_name_offset as usize..];
        let (_, name) = parse_name(i)?;
        Some(name)
    } else {
        None
    };

    let mut sprites = vec![];
    let mut i = &input[..];
    for offset in sprite_offsets.into_iter().map(|x| x as usize) {
        if input.len() < offset {
            return Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
        }
        i = &input[offset..];
        let (_, sprite) = parse_sprite(i)?;
        sprites.push(sprite);
    }

    let mut scripts = BTreeMap::new();
    for (index, offset) in script_offsets.into_iter().map(|(index, offset)| (index as u8, offset as usize)) {
        if input.len() < offset {
            return Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
        }
        i = &input[offset..];
        let mut instruction_offsets = vec![];

        let mut instructions = vec![];
        loop {
            let tell = input.len() - i.len();
            instruction_offsets.push(tell - offset);
            // TODO: maybe check against the size of parsed data?
            let (i2, (time, opcode, _size)) = tuple((le_u16, le_u8, le_u8))(i)?;
            let (i2, instr) = parse_instruction_args(i2, opcode)?;
            instructions.push(Call { time, instr });
            i = i2;
            if opcode == 0 {
                break;
            }
        }
        let mut interrupts = BTreeMap::new();
        let mut j = 0;
        for Call { time: _, instr } in &mut instructions {
            match instr {
                Instruction::Jump(ref mut offset) => {
                    let result = instruction_offsets.binary_search(&(*offset as usize));
                    match result {
                        Ok(ptr) => *offset = ptr as u32,
                        Err(ptr) => {
                            // XXX: use a more specific error instead.
                            return Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Eof)));
                            //println!("Instruction offset not found for pointer: {}", ptr);
                        }
                    }
                }
                Instruction::InterruptLabel(interrupt) => {
                    interrupts.insert(*interrupt, j + 1);
                }
                _ => ()
            }
            j += 1;
        }
        scripts.insert(index, Script {
            instructions,
            interrupts,
        });
    }

    let anm0 = Anm0 {
        size: (width, height),
        format,
        color_key,
        png_filename,
        alpha_filename,
        sprites,
        scripts,
    };
    Ok((i, anm0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Read};
    use std::fs::File;

    #[test]
    fn anm0() {
        let file = File::open("EoSD/CM/player01.anm").unwrap();
        let mut file = io::BufReader::new(file);
        let mut buf = vec![];
        file.read_to_end(&mut buf).unwrap();
        let (_, mut anms) = Anm0::from_slice(&buf).unwrap();
        assert_eq!(anms.len(), 1);
        let anm0 = anms.pop().unwrap();
        assert_eq!(anm0.size, (256, 256));
        assert_eq!(anm0.format, 5);
    }
}
