use touhou_formats::th06::pbg3::PBG3;
use std::env;
use std::path::Path;
use std::fs::{File, create_dir_all};
use std::io::{self, BufReader, Write};

fn main() -> io::Result<()> {
    // Parse arguments.
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <DAT file> <output dir>", args[0]);
        std::process::exit(1);
    }

    let filename = Path::new(&args[1]);
    let output_filename = Path::new(&args[2]);

    let file = File::open(filename)?;
    let file = BufReader::new(file);
    let mut pbg3 = PBG3::from_file(file)?;
    let list = pbg3.list_files().cloned().collect::<Vec<_>>();

    create_dir_all(output_filename)?;
    for filename in list {
        let data = pbg3.get_file(&filename, true)?;
        let mut output = File::create(output_filename.join(filename))?;
        output.write_all(&data)?;
    }

    Ok(())
}
