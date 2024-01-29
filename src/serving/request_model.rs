use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct PredictRequest {
    pub text: String,
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub predictions: Vec<HashMap<&'static str, f32>>,
}
