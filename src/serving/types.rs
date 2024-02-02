use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct PredictRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub predictions: Vec<HashMap<String, f32>>,
}
