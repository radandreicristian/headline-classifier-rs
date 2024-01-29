pub use crate::common::{
    convert_to_array, map_words_to_indices, pad_vector, CategoriesPredictorModel,
};

use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;

pub fn get_predictions(
    text: &str,
    mapping: &Arc<HashMap<&'static str, u32>>,
    model: &Arc<CategoriesPredictorModel>,
) -> Result<Vec<f32>, Box<dyn Error>> {
    let words = text.split_whitespace().collect::<Vec<&str>>();

    let indices = map_words_to_indices(words, mapping);

    let padded_indices = pad_vector::<u32>(indices, 256, 0);

    let padded_indices_array = convert_to_array::<u32, 256>(padded_indices)?;

    let tensor_indices = Tensor::new(&padded_indices_array, &Device::Cpu)?;

    let predictions = model.forward(&tensor_indices)?;

    let predictions_vec = predictions.flatten(0, 1)?.to_vec1()?;

    Ok(predictions_vec)
}

pub fn map_to_class_names_with_scores(
    logits: Vec<f32>,
    class_name_mapping: &HashMap<usize, &'static str>,
    threshold: f32,
) -> Vec<HashMap<&'static str, f32>> {
    let mut class_names_with_logits: Vec<HashMap<&'static str, f32>> = Vec::new();

    for (index, &logit) in logits.iter().enumerate() {
        if logit > threshold {
            if let Some(&class_name) = class_name_mapping.get(&index) {
                let mapping = vec![(class_name, logit)].into_iter().collect();
                class_names_with_logits.push(mapping);
            }
        }
    }

    class_names_with_logits
}
