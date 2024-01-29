use std::vec;

use anyhow::Error;
use candle_core::Tensor;
use polars::io::{csv::CsvReader, SerReader};
#[derive(Clone)]
pub struct Dataset {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
}

fn convert_series_to_string_vector(series: &polars::prelude::Series) -> Result<Vec<String>, Error> {
    let vec_of_strings: Vec<String> = series
        .str()?
        .into_iter()
        .map(|optional| match optional {
            Some(val) => val.to_string(),
            None => "None".to_string(),
        })
        .collect();

    Ok(vec_of_strings)
}

pub fn read_data(path: &str) -> Result<(Vec<String>, Vec<String>), Error> {
    let df = CsvReader::from_path(path)?.has_header(true).finish()?;

    let data_series = df.column("text")?;

    let labels_series = df.column("labels")?;

    let data = convert_series_to_string_vector(&data_series)?;

    let labels = convert_series_to_string_vector(&labels_series)?;

    Ok((data, labels))
}
