//! Development diagnostic: mono 16 kHz float32-le PCM -> frontend feature file.
use std::{io::Write, path::PathBuf};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: gemma_frontend INPUT.f32 OUTPUT.f32".into());
    }
    let bytes = std::fs::read(&args[0])?;
    if bytes.len() % 4 != 0 {
        return Err("PCM bytes must be a multiple of four".into());
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let f = grain_elevator::gemma::features(&samples)?;
    let mut output = std::fs::File::create(PathBuf::from(&args[1]))?;
    for value in f.values {
        output.write_all(&value.to_le_bytes())?;
    }
    println!(
        "{}",
        serde_json::json!({"frames":f.frames,"mask":f.mask,"tokens":f.tokens})
    );
    Ok(())
}
