#![cfg(feature = "cli")]

mod catboost;
mod treelite;

use std::{
    fmt, fs,
    io::{self, Write},
    path,
};

use byteorder::{LittleEndian, WriteBytesExt};
use clap::{arg, command, value_parser};

trait Fail<T> {
    fn fail<'a>(self, message: impl FnOnce() -> &'a str) -> T;
}

impl<T, E: fmt::Display> Fail<T> for Result<T, E> {
    fn fail<'a>(self, message: impl FnOnce() -> &'a str) -> T {
        self.unwrap_or_else(|err| {
            eprintln!("{}: {}", message(), err);
            std::process::exit(1);
        })
    }
}

fn main() {
    let matches = command!()
        .about("Compile a tree-based machine learning model to an ACAM map encoded in the NumPy format")
        .arg_required_else_help(true)
        .arg(
            arg!(-t --type <MODEL_TYPE> "The type of model that is passed")
                .value_parser(["treelite", "catboost"])
                .default_value("treelite"),
        )
        .arg(
            arg!(<INPUT_FILE>)
                .required(true)
                .value_parser(value_parser!(path::PathBuf))
                .help("Path to the model JSON dump"),
        )
        .arg(
            arg!(<OUTPUT_FILE>)
                .required(true)
                .value_parser(value_parser!(path::PathBuf))
                .help("Path to the resulting NumPy file"),
        )
        .get_matches();

    let model_type: &String = matches.get_one("type").unwrap();
    let input_path: &path::PathBuf = matches.get_one("INPUT_FILE").unwrap();
    let output_path: &path::PathBuf = matches.get_one("OUTPUT_FILE").unwrap();

    let json = fs::read_to_string(input_path).fail(|| "could not read input file");

    let (compiled_model, shape) = match model_type.as_str() {
        "treelite" => {
            eprintln!("loading model...");
            let model = treelite::Model::from_json(&json).fail(|| "failed to deserialize model");
            eprintln!("compiling model...");
            treelite::compile_model(&model)
        }
        "catboost" => {
            eprintln!("loading model...");
            let model = catboost::Model::from_json(&json).fail(|| "failed to deserialize model");
            eprintln!("compiling model...");
            catboost::compile_model(&model)
        }
        _ => unreachable!(),
    };

    eprintln!("saving output...");
    write_compiled_model(output_path, compiled_model, shape)
        .fail(|| "could not write compiled model");

    eprintln!("done!")
}

fn write_compiled_model(
    path: &path::PathBuf,
    model: Vec<f64>,
    shape: (usize, usize),
) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = io::BufWriter::new(file);

    writer.write_all(b"\x93NUMPY")?; // numpy magic bytes
    writer.write_u8(1)?; // major version
    writer.write_u8(0)?; // minor version

    // numpy metadata header
    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
        shape.0, shape.1
    );

    // write header length (including newline)
    writer.write_u16::<LittleEndian>(dict.len() as u16 + 1)?;

    // write header and newline
    writer.write_all(dict.as_bytes())?;
    writer.write_u8(b'\n')?;

    // write the actual data
    for value in model {
        writer.write_f64::<LittleEndian>(value)?;
    }

    Ok(())
}
