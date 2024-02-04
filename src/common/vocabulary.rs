use super::exception::VocabularyLoadError;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::File,
    io::{Read, Write},
};

#[derive(Serialize, Deserialize, Debug)]
struct Vocabulary {
    vocabulary: Vec<String>,
}

/// Creates a vocabulary from a given corpus of sentences.
///
/// The function takes a reference to a vector of strings representing a corpus
/// of sentences and returns a vector of unique words found in the corpus. Words
/// are separated by whitespace in each sentence.
///
/// # Arguments
///
/// * `corpus` - A reference to a vector of strings containing sentences.
///
/// # Returns
///
/// A vector of unique words found in the corpus.
pub fn make_vocabulary(corpus: &Vec<String>) -> Vec<String> {
    let mut vocabulary: HashSet<String> = HashSet::new();

    let punctuation_regex = Regex::new(r"[[:punct:]]").unwrap();

    for sentence in corpus {
        let sentence_without_punctuation = punctuation_regex.replace_all(sentence, "");

        let words: Vec<&str> = sentence_without_punctuation.split_whitespace().collect();

        for word in words {
            vocabulary.insert(word.to_string());
        }
    }
    vocabulary.into_iter().collect::<Vec<String>>()
}

/// Load a vocabulary from a JSON file.
///
/// # Arguments
///
/// * `file_path` - A string containing the path to the JSON file from which the vocabulary will be loaded.
///
/// # Errors
///
/// This function can return an error of type `VocabularyLoadError` if there are issues with file opening, reading, JSON deserialization, or data conversion.
///
/// # Returns
///
/// This function returns a `Result<Vec<String>, VocabularyLoadError>`, where `Vec<String>` represents the loaded vocabulary on success, and `VocabularyLoadError` represents any encountered errors specific to vocabulary loading
pub fn load_vocabulary(file_path: &str) -> Result<Vec<String>, VocabularyLoadError> {
    let mut file = File::open(file_path)?;
    let mut json_data = String::new();

    file.read_to_string(&mut json_data)?;

    let vocabulary: Vocabulary = serde_json::from_str(&json_data)?;

    Ok(vocabulary.vocabulary)
}

/// Store a vocabulary in a JSON file.
///
/// # Arguments
///
/// * `vocabulary` - A reference to a Vec<String> containing the vocabulary to be stored.
/// * `file_path` - A string containing the path to the JSON file where the vocabulary will be stored.
///
/// # Errors
///
/// This function can return an error of type `anyhow::Error` if there are issues with file creation, JSON serialization, or file writing.
///
/// # Returns
///
/// This function returns `Result<(), anyhow::Error>`, where `()` indicates success, and `anyhow::Error` represents any encountered errors.
pub fn store_vocabulary(vocabulary: &Vec<String>, file_path: &str) -> Result<(), anyhow::Error> {
    let mut file = File::create(file_path)?;
    file.write_all(
        serde_json::to_string(&Vocabulary {
            vocabulary: vocabulary.to_owned(),
        })?
        .as_bytes(),
    )?;
    Ok(())
}
