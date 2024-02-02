use anyhow::{anyhow, Error};
use candle_core::{Device, Tensor};
use candle_nn::ops::sigmoid;
use common::{map_to_indices, pad_vector, CategoriesPredictorModel, MAX_SEQ_LEN};
use std::collections::HashMap;

pub fn get_predictions(
    text: &str,
    word_to_index: &HashMap<String, u32>,
    model: &CategoriesPredictorModel,
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
