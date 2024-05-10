use std::io;

use bmc::*;

fn main() -> io::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();

    if args.len() < 2 {
        println!("Usage: {} FILENAME", args[0]);
        std::process::exit(1);
    }

    let source = std::fs::read_to_string(&args[1])?;
    let tokens = scan(&source);
    println!("{tokens:#?}");
    Ok(())
}
