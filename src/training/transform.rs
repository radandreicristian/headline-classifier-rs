use std::collections::HashMap;
use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use common::{map_words_to_indices, pad_vector};

pub fn encode(data: &Vec<String>, max_seq_len: usize, vocabulary_index_mapping: &HashMap<String, u32>, device: &Device) -> Result<Tensor, Error>{
    let indices: Vec<u32> = data
    .iter()
    .flat_map(|sentence| {
        let words: Vec<String> = sentence.split_whitespace().map(|s| s.to_string()).collect();
        let indices = map_words_to_indices(words, vocabulary_index_mapping);
        pad_vector(indices, max_seq_len, 0)
    })
    .collect();

    let tensor = Tensor::from_vec(
        indices.clone(),
        (max_seq_len, indices.len() / max_seq_len),
        &device,
    )?;

    Ok(tensor)
}