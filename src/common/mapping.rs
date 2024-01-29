use std::{collections::HashMap, hash::Hash, ops::Index};

pub fn create_vocabulary_to_index_mapping(vocabulary: &Vec<String>) -> HashMap<String, u32> {
    let mut vocab_to_index = HashMap::<String, u32>::new();

    vocab_to_index.insert("<UNK>".to_string(), 0);

    for (index, word) in vocabulary.iter().enumerate() {
        vocab_to_index.insert(word.clone(), index as u32 + 1);
    }

    vocab_to_index
}

pub fn map_words_to_indices(words: Vec<String>, mapping: &HashMap<String, u32>) -> Vec<u32> {
    words
        .iter()
        .map(|word| mapping.get(word).copied().unwrap_or(0))
        .collect()
}

pub fn create_class_mappings_from_class_names(
    class_names: Vec<String>,
) -> (HashMap<usize, String>, HashMap<String, usize>) {
    let mut index_to_class: HashMap<usize, String> = HashMap::new();
    let mut class_to_index: HashMap<String, usize> = HashMap::new();

    for (index, word) in class_names.iter().enumerate() {
        index_to_class.insert(index, word.to_string());
        class_to_index.insert(word.to_string(), index);
    }

    (index_to_class, class_to_index)
}

/// Create a class mapping from the labels of the trainig set.
///
/// This function assumes that labels is a vector of strings, where each string can represent either no classes (empty string), a single class, or multiple classes (delimited by comma).
///
///
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

pub fn multi_hot_encode(labels: Vec<String>, class_to_index: &HashMap<String, u32>) -> Vec<u32> {
    let n_classes = class_to_index.len();
    let mut all_encodings: Vec<u32> = Vec::new(); // Initialize a single encodings vector

    for label in labels {
        let mut label_encodings = vec![0u32; n_classes];
        log::debug!("Label: {:?}", label);
        if label == "" {
            log::debug!("Encoding: {:?}", label_encodings);
            all_encodings.append(&mut label_encodings);
            continue; // Skip empty labels
        }

        let label_classes: Vec<&str> = label.split('|').collect();
        let label_class_indices: Vec<u32> = label_classes
            .iter()
            .map(|label| {
                *class_to_index.get(&label.to_string()).unwrap() // Default to 0 if label not found
            })
            .collect();

        for &index in &label_class_indices {
            label_encodings[index as usize] = 1u32;
        }
        log::debug!("Encoding: {:?}", label_encodings);

        all_encodings.append(&mut label_encodings);
    }
    log::debug!("All encodings {:?}", all_encodings);
    all_encodings
}
