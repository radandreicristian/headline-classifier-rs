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

/// Convert a Polars Series into a vector of strings.
///
/// # Arguments
///
/// * `series` - A reference to a Polars Series that is to be converted.
///
/// # Errors
///
/// This function can return an error if there are issues with type conversion or data manipulation.
///
/// # Returns
///
/// This function returns a `Result<Vec<String>, Error>`, where `Vec<String>` represents the converted strings on success, and `Error` represents any encountered errors.
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

/// Read data from a CSV file and extract text and labels.
///
/// # Arguments
///
/// * `path` - A string containing the path to the CSV file to be read.
///
/// # Errors
///
/// This function can return an error if there are issues with CSV file reading, data extraction, or type conversion.
///
/// # Returns
///
/// This function returns a `Result<(Vec<String>, Vec<String>), Error>`, where the tuple represents the extracted text and labels as vectors of strings on success, and `Error` represents any encountered errors.
pub fn read_data(path: &str) -> Result<(Vec<String>, Vec<String>), Error> {
    let df = CsvReader::from_path(path)?.has_header(true).finish()?;

    let data_series = df.column("text")?;

    let labels_series = df.column("labels")?;

    let data = convert_series_to_string_vector(&data_series)?;

    let labels = convert_series_to_string_vector(&labels_series)?;

    Ok((data, labels))
}
