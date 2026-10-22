use grain_elevator::{payload::Payload, synth, Encoder};
use std::{io::Write, path::PathBuf};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    match cmd.as_str() {
        "info" => {
            let p=Payload::open(&std::env::current_exe()?)?;
            println!("{}",serde_json::to_string_pretty(&p.info)?);
        },
        "embed" => {
            let path=PathBuf::from(args.next().ok_or("usage: grain-elevator embed AUDIO [--output FILE]")?);
            let output=match args.next().as_deref(){Some("--output")=>Some(PathBuf::from(args.next().ok_or("missing output path")?)),None=>None,_=>return Err("unknown embed option".into())};
            if args.next().is_some(){return Err("unexpected argument".into());}
            let enc=Encoder::open()?;
            let value=enc.embed_file(path)?;
            if let Some(path)=output {
                if path.with_extension("metadata.json")==path {return Err("output and metadata paths must differ".into());}
                let mut file=std::fs::OpenOptions::new().write(true).create_new(true).open(&path)?;
                for v in &value.values {file.write_all(&v.to_le_bytes())?;}
                let metadata=serde_json::json!({"identity":value.identity,"shape":value.shape,"mask":value.mask,"encoding":"float32-le","count":value.values.len()});
                let mut sidecar=std::fs::OpenOptions::new().write(true).create_new(true).open(path.with_extension("metadata.json"))?;
                sidecar.write_all(&serde_json::to_vec_pretty(&metadata)?)?;
                println!("{} floats; shape {:?}",value.values.len(),value.shape);
            } else { println!("identity: {}\nshape: {:?}\nmask: {:?}\n{:?}",value.identity,value.shape,value.mask,value.values); }
        },
        "synth-test" => {
            let output=match args.next().as_deref(){Some("--report")=>Some(PathBuf::from(args.next().ok_or("missing report path")?)),None=>None,_=>return Err("unknown synth-test option".into())};
            let report=synth::measure(&Encoder::open()?)?;
            for case in &report.cases {println!("{} {:?}: d12={:.6}, d23={:.6}, d13={:.6} {}",case.knob,case.values,case.distances[0],case.distances[1],case.distances[2],if case.adjacent_12_within_span&&case.adjacent_23_within_span{"PASS"}else{"VIOLATION"});}
            println!("{}/{} monotonic triples ({:.1}%). Human listening remains the final judgment.",report.passed,report.total,100.0*report.pass_fraction);
            if let Some(path)=output {std::fs::write(path,serde_json::to_vec_pretty(&report)?)?;}
            // This is a measurement, not a proof that a learned model must be monotonic.
        },
        "help"|"--help"|"-h" => println!("grain-elevator info\ngrain-elevator embed AUDIO [--output FLOAT32_FILE]\ngrain-elevator synth-test [--report JSON_FILE]"),
        _=>return Err("unknown command; use --help".into()),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
