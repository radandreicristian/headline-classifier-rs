use std::{error::Error, fmt};

#[derive(Debug)]
pub struct MultiHotEncodeError {
    pub message: String,
}

impl MultiHotEncodeError {
    pub fn new(message: &str) -> MultiHotEncodeError {
        MultiHotEncodeError {
            message: message.to_string(),
        }
    }
}

impl fmt::Display for MultiHotEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for MultiHotEncodeError {}

#[derive(Debug)]
pub enum VocabularyLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl From<std::io::Error> for VocabularyLoadError {
    fn from(err: std::io::Error) -> Self {
        VocabularyLoadError::Io(err)
    }
}

impl From<serde_json::Error> for VocabularyLoadError {
    fn from(err: serde_json::Error) -> Self {
        VocabularyLoadError::Json(err)
    }
}

impl std::error::Error for VocabularyLoadError {}

impl std::fmt::Display for VocabularyLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VocabularyLoadError::Io(err) => write!(f, "IO error: {}", err),
            VocabularyLoadError::Json(err) => write!(f, "JSON error: {}", err),
        }
    }
}
