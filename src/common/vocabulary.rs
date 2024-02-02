use std::{collections::HashSet, fs::File, io::Read};
use regex::Regex;
use serde::Deserialize;
use super::exception::VocabularyLoadError;

#[derive(Deserialize, Debug)]
struct Vocabulary {
    vocabulary: Vec<String>
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

pub fn load_vocabulary(file_path: &str) -> Result<Vec<String>, VocabularyLoadError> {
    let mut file = File::open(file_path)?;
    let mut json_data = String::new();

    file.read_to_string(&mut json_data)?;

    let vocabulary: Vocabulary = serde_json::from_str(&json_data)?;

    Ok(vocabulary.vocabulary)
}
