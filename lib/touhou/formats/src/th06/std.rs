//! STD background format support.

use nom::{
    IResult,
    bytes::complete::tag,
    number::complete::{le_u8, le_u16, le_u32, le_i32, le_f32},
    sequence::tuple,
    combinator::map,
    multi::{many0, count},
    error::ErrorKind,
    Err,
};
use encoding_rs::SHIFT_JIS;

/// A float position in the 3D space.
#[derive(Debug, Clone)]
pub struct Position {
    /// X component.
    pub x: f32,

    /// Y component.
    pub y: f32,

    /// Z component.
    pub z: f32,
}

/// A 3D box around something.
#[derive(Debug, Clone)]
struct Box3D {
    width: f32,
    height: f32,
    depth: f32,
}

/// A 2D box around something.
#[derive(Debug, Clone)]
pub struct Box2D {
    /// Width.
    pub width: f32,

    /// Height.
    pub height: f32,
}

/// A quad in the 3D space.
#[derive(Debug, Clone)]
pub struct Quad {
    /// The anm script to run for this quad.
    pub anm_script: u16,

    /// The position of this quad in the 3D space.
    pub pos: Position,

    /// The size of this quad.
    pub size_override: Box2D,
}

/// A model formed of multiple quads in space.
#[derive(Debug, Clone)]
pub struct Model {
    /// TODO: find what that is.
    pub unknown: u16,

    /// The bounding box around this model.
    pub bounding_box: [f32; 6],

    /// The quads composing this model.
    pub quads: Vec<Quad>,
}

/// An instance of a model.
#[derive(Debug, Clone)]
pub struct Instance {
    /// The instance identifier.
    pub id: u16,

    /// Where to position the instance of this model.
    pub pos: Position,
}

/// A single instruction, part of a `Script`.
#[derive(Debug, Clone)]
pub struct Call {
    /// Time at which this instruction will be called.
    pub time: u32,

    /// The instruction to call.
    pub instr: Instruction,
}

/// Parse a SHIFT_JIS byte string of length 128 into a String.
#[allow(non_snake_case)]
pub fn le_String(i: &[u8]) -> IResult<&[u8], String> {
    let data = i.splitn(2, |c| *c == b'\0').nth(0).unwrap();
    let (string, _encoding, _replaced) = SHIFT_JIS.decode(data);
    Ok((&i[128..], string.into_owned()))
}

/// Main struct of the STD stage format.
#[derive(Debug, Clone)]
pub struct Stage {
    /// The name of the stage.
    pub name: String,

    /// A list of (name, path) of background music.
    // TODO: there are maximum four of them, and in practice never more than 2.
    pub musics: Vec<Option<(String, String)>>,

    /// List of models.
    pub models: Vec<Model>,

    /// List of instances.
    pub instances: Vec<Instance>,

    /// List of instructions in the script.
    pub script: Vec<Call>,
}

impl Stage {
    /// Parse a slice of bytes into an `Stage` struct.
    pub fn from_slice(data: &[u8]) -> IResult<&[u8], Stage> {
        parse_stage(data)
    }
}

macro_rules! declare_stage_instructions {
    ($($opcode:tt => fn $name:ident($($arg:ident: $arg_type:ident),*)),*,) => {
        /// Available instructions in an `Stage`.
        #[allow(missing_docs)]
        #[derive(Debug, Clone, Copy)]
        pub enum Instruction {
            $(
                $name($($arg_type),*)
            ),*
        }

        fn parse_instruction_args(input: &[u8], opcode: u16) -> IResult<&[u8], Instruction> {
            let mut i = &input[..];
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
                _ => unreachable!()
            };
            Ok((i, instr))
        }
    };
}

declare_stage_instructions!{
    0 => fn SetViewpos(x: f32, y: f32, z: f32),
    1 => fn SetFog(r: u8, g: u8, b: u8, a: u8, near: f32, far: f32),
    2 => fn SetViewpos2(x: f32, y: f32, z: f32),
    3 => fn StartInterpolatingViewpos2(frame: u32, _unused: i32, _unused: i32),
    4 => fn StartInterpolatingFog(frame: u32, _unused: i32, _unused: i32),
    5 => fn Unknown(_unused: i32, _unused: i32, _unused: i32),
}

fn parse_quad(i: &[u8]) -> IResult<&[u8], Quad> {
    let (i, (unk1, size)) = tuple((le_u16, le_u16))(i)?;
    if unk1 == 0xffff {
        return Err(Err::Error(nom::error::Error::new(i, ErrorKind::Eof)));
    }
    // TODO: replace this assert with a custom error.
    assert_eq!(size, 0x1c);
    let (i, (anm_script, _, x, y, z, width, height)) = tuple((le_u16, tag(b"\0\0"), le_f32, le_f32, le_f32, le_f32, le_f32))(i)?;
    let quad = Quad {
        anm_script,
        pos: Position { x, y, z },
        size_override: Box2D { width, height },
    };
    Ok((i, quad))
}

fn parse_model(i: &[u8]) -> IResult<&[u8], Model> {
    let (i, (_id, unknown, x, y, z, width, height, depth, quads)) = tuple((le_u16, le_u16, le_f32, le_f32, le_f32, le_f32, le_f32, le_f32, many0(parse_quad)))(i)?;
    let bounding_box = [x, y, z, width, height, depth];
    let model = Model {
        unknown,
        bounding_box,
        quads,
    };
    Ok((i, model))
}

fn parse_instance(i: &[u8]) -> IResult<&[u8], Instance> {
    let (i, (id, unknown, x, y, z)) = tuple((le_u16, le_u16, le_f32, le_f32, le_f32))(i)?;
    if id == 0xffff && unknown == 0xffff {
        return Err(Err::Error(nom::error::Error::new(i, ErrorKind::Eof)));
    }
    // TODO: replace this assert with a custom error.
    assert_eq!(unknown, 0x100);
    let instance = Instance {
        id,
        pos: Position { x, y, z },
    };
    Ok((i, instance))
}

fn parse_instruction(i: &[u8]) -> IResult<&[u8], Call> {
    let (i, (time, opcode, size)) = tuple((le_u32, le_u16, le_u16))(i)?;
    if time == 0xffffffff && opcode == 0xffff && size == 0xffff {
        return Err(Err::Error(nom::error::Error::new(i, ErrorKind::Eof)));
    }
    // TODO: replace this assert with a custom error.
    assert_eq!(size, 12);
    let (i, instr) = parse_instruction_args(i, opcode)?;
    println!("{} {:?}", time, instr);
    let call = Call { time, instr };
    Ok((i, call))
}

fn parse_stage(input: &[u8]) -> IResult<&[u8], Stage> {
    let i = &input[..];

    let (i, (num_models, _num_faces, object_instances_offset, script_offset, _, name, music_names, music_paths)) = tuple((
        le_u16, le_u16, le_u32, le_u32, tag(b"\0\0\0\0"),
        le_String,
        map(tuple((le_String, le_String, le_String, le_String)), |(a, b, c, d)| [a, b, c, d]),
        map(tuple((le_String, le_String, le_String, le_String)), |(a, b, c, d)| [a, b, c, d])
    ))(i)?;
    let musics = music_names.iter().zip(&music_paths).map(|(name, path)| if name == " " { None } else { Some((name.clone(), path.clone())) }).collect();

    let (_, offsets) = count(le_u32, num_models as usize)(i)?;

    let mut models = vec![];
    for offset in offsets {
        let (_, model) = parse_model(&input[offset as usize..])?;
        models.push(model);
    }

    let (_, instances) = many0(parse_instance)(&input[object_instances_offset as usize..])?;
    let (_, script) = many0(parse_instruction)(&input[script_offset as usize..])?;

    let stage = Stage {
        name,
        musics,
        models,
        instances,
        script,
    };
    Ok((b"", stage))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Read};
    use std::fs::File;

    #[test]
    fn std() {
        let file = File::open("EoSD/ST/stage1.std").unwrap();
        let mut file = io::BufReader::new(file);
        let mut buf = vec![];
        file.read_to_end(&mut buf).unwrap();
        let (_, stage) = Stage::from_slice(&buf).unwrap();
        assert_eq!(stage.name, "夢幻夜行絵巻　～ Mystic Flier");
        assert_eq!(stage.musics.len(), 4);
        assert_eq!(stage.models.len(), 13);
        assert_eq!(stage.instances.len(), 90);
        assert_eq!(stage.script.len(), 21);
    }
}
