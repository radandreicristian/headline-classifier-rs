use std::{error::Error, fmt};

#[derive(Debug)]
pub enum InferenceError {
    ArrayConversionError(&'static str),
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            InferenceError::ArrayConversionError(error_message) => {
                write!(f, "Inference Error - {}", error_message)
            }
        }
    }
}

impl Error for InferenceError {}
