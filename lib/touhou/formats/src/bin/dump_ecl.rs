use touhou_formats::th06::ecl::{Ecl, CallMain, CallSub, Rank};
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::{self, BufReader, Read};

pub fn load_file_into_vec<P: AsRef<Path>>(filename: P) -> io::Result<Vec<u8>> {
    let file = File::open(filename)?;
    let mut file = BufReader::new(file);
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

fn format_rank(rank: &Rank) -> String {
    format!("{}{}{}{}", if rank.contains(Rank::EASY) { 'E' } else { ' ' },
                        if rank.contains(Rank::NORMAL) { 'N' } else { ' ' },
                        if rank.contains(Rank::HARD) { 'H' } else { ' ' },
                        if rank.contains(Rank::LUNATIC) { 'L' } else { ' ' })
}

fn print_sub_instruction(call: &CallSub) {
    let CallSub { time, rank_mask, param_mask: _, instr } = call;
    println!("    {:>5}: {}: {:?}", time, format_rank(rank_mask), instr);
}

fn print_main_instruction(call: &CallMain) {
    let CallMain { time, sub, instr } = call;
    println!("    {:>5}: sub {:>2}: {:?}", time, sub, instr);
}

fn main() {
    // Parse arguments.
    let args: Vec<_> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <ECL file>", args[0]);
        return;
    }
    let ecl_filename = Path::new(&args[1]);

    // Open the ECL file.
    let buf = load_file_into_vec(ecl_filename).unwrap();
    let (_, ecl) = Ecl::from_slice(&buf).unwrap();

    for (i, main) in ecl.mains.iter().enumerate() {
        println!("Main {} {{", i);
        for call in main.instructions.iter() {
            print_main_instruction(call);
        }
        println!("}}");
        println!();
    }

    for (i, sub) in ecl.subs.iter().enumerate() {
        println!("Sub {} {{", i);
        for call in sub.instructions.iter() {
            print_sub_instruction(call);
        }
        println!("}}");
        println!();
    }
}
