use std::{collections::HashMap, fs::File, io::{Read, Write}};
use serde::{Deserialize, Serialize};
use anyhow::Error;
use super::MultiHotEncodeError;

/// Creates a mapping from vocabulary words to their corresponding indices.
///
/// This function takes a reference to a vector of strings `vocabulary` and creates a `HashMap`
/// where each unique word in the vocabulary is associated with its index (a `u32` value). The
/// resulting mapping is used for tasks such as converting text data into sequences of indices.
///
/// The special token "<UNK>" is inserted at index 0 in the mapping to represent any unknown words.
/// When using this mapping to convert words to indices, if a word is not found in the vocabulary,
/// it is mapped to the index of "<UNK>" (index 0).
///
/// # Arguments
///
/// * `vocabulary`: A reference to a vector of strings representing the vocabulary.
///
/// # Returns
///
/// A `HashMap` mapping words to their corresponding indices.
pub fn create_vocabulary_to_index_mapping(vocabulary: &Vec<String>) -> HashMap<String, u32> {
    let mut vocab_to_index = HashMap::<String, u32>::new();

    // Special token for any unknown words. In map_to_indices, unwrap_or(0) is used to map any unknown words to this token index.
    vocab_to_index.insert("<UNK>".to_string(), 0);

    for (index, word) in vocabulary.iter().enumerate() {
        vocab_to_index.insert(word.clone(), index as u32 + 1);
    }

    vocab_to_index
}

/// Maps a vector of words to their corresponding indices using a mapping.
///
/// This function takes a vector of strings `words` and a reference to a mapping `mapping`, which
/// associates words with their corresponding indices. It maps each word in the input vector to its
/// index using the provided mapping. If a word is not found in the mapping, it is mapped to 0 by
/// default.
///
/// # Arguments
///
/// * `words`: A vector of strings representing the words to be mapped to indices.
/// * `mapping`: A reference to a `HashMap` mapping words to their corresponding indices.
///
/// # Returns
///
/// A vector of `u32` values representing the indices of the input words based on the provided mapping.
pub fn map_to_indices(words: Vec<String>, mapping: &HashMap<String, u32>) -> Vec<u32> {
    words
        .iter()
        .map(|word| mapping.get(word).copied().unwrap_or(0))
        .collect()
}

/// Creates mappings between class labels and their corresponding indices.
///
/// This function takes a reference to a vector of class labels `labels` and creates two mappings:
///   - A mapping from class labels (strings) to their corresponding indices (u32).
///   - A reverse mapping from indices to class labels.
///
/// Class labels are often represented as strings separated by '|' characters, allowing multiple
/// labels to be associated with a single data point. This function processes the labels and assigns
/// unique indices to each unique label encountered.
///
/// # Arguments
///
/// * `labels`: A reference to a vector of strings representing class labels.
///
/// # Returns
///
/// A tuple containing two `HashMap` instances:
///   - The first `HashMap` maps class labels (strings) to their corresponding indices (u32).
///   - The second `HashMap` maps indices to their corresponding class labels.
pub fn create_class_mapping_from_labels(
    labels: &Vec<String>,
) -> (HashMap<String, u32>, HashMap<u32, String>) {
    let mut class_to_index: HashMap<String, u32> = HashMap::new();
    let mut index_to_class: HashMap<u32, String> = HashMap::new();

    let mut n_classes = 0;

    for word in labels.iter() {
        if word == "" {
            continue;
        }
        let labels: Vec<&str> = word.split('|').collect();

        for label in labels {
            if !class_to_index.contains_key(label) {
                class_to_index.insert(label.to_string(), n_classes);
                index_to_class.insert(n_classes, label.to_string());
                n_classes += 1;
            }
        }
    }

    (class_to_index, index_to_class)
}

/// Converts a list of labels into a multi-hot encoding using the provided mapping.
///
/// Given a list of labels and a mapping of class names to their respective indices,
/// this function generates a multi-hot encoding where each label corresponds to a binary vector.
/// If a label is not found in the mapping, an error is returned.
///
/// # Arguments
///
/// * `labels` - A vector of labels to be encoded.
/// * `class_to_index` - A reference to a `HashMap` containing class names as keys and their
///   corresponding indices as values.
///
/// # Errors
///
/// If a label in `labels` is not found in `class_to_index`, an `Err` variant is returned
/// with an associated error message indicating which label was not found.
///
/// # Returns
///
/// Returns a `Result` where:
/// - `Ok(encodings)` contains the multi-hot encodings as a vector of u32 values.
/// - `Err(err)` contains a `MultiHotEncodeError` with a description of the error.
pub fn multi_hot_encode(
    labels: Vec<String>,
    class_to_index: &HashMap<String, u32>,
) -> Result<Vec<u32>, MultiHotEncodeError> {
    let n_classes = class_to_index.len();
    let mut all_encodings: Vec<u32> = Vec::new(); // Initialize a single encodings vector

    for label in labels {
        let mut label_encodings = vec![0u32; n_classes];
        log::debug!("Label: {:?}", label);
        if label == "" {
            log::debug!("Encoding: {:?}", label_encodings);
            all_encodings.append(&mut label_encodings);
            // Skip empty labels
            continue;
        }

        let label_classes: Vec<&str> = label.split('|').collect();
        for label_class in label_classes {
            if let Some(&index) = class_to_index.get(&label_class.to_string()) {
                label_encodings[index as usize] = 1u32;
            } else {
                return Err(MultiHotEncodeError::new(&format!("Label not found: {}", label_class)));
            }
        }
        log::debug!("Encoding: {:?}", label_encodings);

        all_encodings.append(&mut label_encodings);
    }
    log::debug!("All encodings {:?}", all_encodings);

    Ok(all_encodings)
}


#[derive(Serialize, Deserialize, Debug)]
struct IndexToClassMapping {
    mapping: HashMap<u32, String>
}

pub fn store_index_to_class_mapping(index_to_class: &HashMap<u32, String>, file_path: &str) -> Result<(), Error> {
    let mut file = File::create(file_path)?;
    file.write_all(serde_json::to_string(&IndexToClassMapping{mapping: index_to_class.to_owned()})?.as_bytes())?;
    Ok(())
}

pub fn load_index_to_class_mapping(file_path: &str) -> Result<HashMap<u32, String>, Error> {
    let mut file = File::open(file_path)?;
    let mut json_data = String::new();

    file.read_to_string(&mut json_data)?;

    let mapping: IndexToClassMapping = serde_json::from_str(&json_data)?;

    Ok(mapping.mapping)
}