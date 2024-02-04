use anyhow::{anyhow, Error};
use candle_core::{Device, Tensor};
use candle_nn::ops::sigmoid;
use common::{map_to_indices, pad_vector, HeadlineClassifierModel, MAX_SEQ_LEN};
use std::collections::HashMap;


/// Get predictions from a headline classification model for the given text.
///
/// # Arguments
///
/// * `text` - A string containing the input text for which predictions are to be generated.
/// * `word_to_index` - A reference to a HashMap<String, u32> mapping words to their corresponding indices.
/// * `model` - A reference to a HeadlineClassifierModel used for making predictions.
///
/// # Errors
///
/// This function can return an error if there are issues with tokenization, index mapping, tensor conversion, model inference, or sigmoid transformation.
///
/// # Returns
///
/// This function returns a `Result<Vec<f32>, Error>`, where `Vec<f32>` represents the predicted probabilities for each class on success, and `Error` represents any encountered errors.
pub fn get_predictions(
    text: &str,
    word_to_index: &HashMap<String, u32>,
    model: &HeadlineClassifierModel,
) -> Result<Vec<f32>, Error> {
    let words = text
        .split_whitespace()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

    let indices = map_to_indices(words, word_to_index);

    let padded_indices: Vec<u32> = pad_vector(indices, MAX_SEQ_LEN, 0);

    let padded_indices_array: [u32; MAX_SEQ_LEN] = match padded_indices.try_into() {
        Ok(array) => Ok(array),
        Err(_) => Err(anyhow!("Failed to convert Vec<u32> to [u32; _]")),
    }?;

    let tensor_indices = Tensor::new(&padded_indices_array, &Device::Cpu)?;

    let predictions = model.forward(&tensor_indices)?;

    let predictions_vec = sigmoid(&predictions)?.flatten(0, 1)?.to_vec1()?;

    Ok(predictions_vec)
}

/// Map logits to class names with scores based on a given threshold.
///
/// # Arguments
///
/// * `logits` - A vector of f32 representing the logits or scores for each class.
/// * `index_to_class` - A reference to a HashMap<u32, String> mapping class indices to their corresponding names.
/// * `threshold` - A threshold value used to filter out class names with logits below this value.
///
/// # Returns
///
/// This function returns a vector of HashMaps, where each HashMap associates class names (String) with their corresponding scores (f32).
pub fn map_to_class_names_with_scores(
    logits: Vec<f32>,
    index_to_class: &HashMap<u32, String>,
    threshold: f32,
) -> Vec<HashMap<String, f32>> {
    let mut class_names_with_logits: Vec<HashMap<String, f32>> = Vec::new();

    for (index, &logit) in logits.iter().enumerate() {
        if logit > threshold {
            if let Some(class_name) = index_to_class.get(&(index as u32)) {
                let mapping = vec![(class_name, logit)]
                    .into_iter()
                    .map(|(c, l)| (c.to_string(), l))
                    .collect();
                class_names_with_logits.push(mapping);
            }
        }
    }

    class_names_with_logits
}
