use anyhow::{Ok, Error};
use polars::{error::PolarsError, io::SerReader, prelude::CsvReader};

pub fn main() -> Result<(), Error> {
    let df = CsvReader::from_path("./data/train.csv")?.has_header(true).finish()?;

    let series = df.column("text").unwrap().to_owned();

    let vec_of_strings: Vec<&str> = series.str().unwrap().into_iter().map(|optional| {
        let x = match optional {
            Some(val) => val, None => "fk"
        };
        x
    }).collect();

    println!("{:?}", vec_of_strings);


    Ok(())
}