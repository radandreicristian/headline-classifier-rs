use candle_core::{DType, Device};
use common::{self, PREDICTION_THRESHOLD};
mod inference;
mod types;

use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;
use std::sync::Arc;

use common::{
    create_vocabulary_to_index_mapping, load_index_to_class_mapping, load_vocabulary,
    CategoriesPredictorModel, ModelConfig, INDEX_TO_CLASS_PATH, MODEL_PATH, VOCAB_PATH,
};
use inference::{get_predictions, map_to_class_names_with_scores};
use types::{PredictRequest, PredictResponse};
use warp::Filter;

fn with_shared_data(
    shared_data: SharedData,
) -> impl Filter<Extract = (SharedData,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || shared_data.clone())
}

#[derive(Clone)]
struct SharedData {
    word_to_index: Arc<HashMap<String, u32>>,
    index_to_class: Arc<HashMap<u32, String>>,
    model: Arc<CategoriesPredictorModel>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    // Load the vocabulary and the mappings
    let vocabulary = load_vocabulary(VOCAB_PATH)?;

    let word_to_index = Arc::new(create_vocabulary_to_index_mapping(&vocabulary));
    let index_to_class = load_index_to_class_mapping(INDEX_TO_CLASS_PATH)?;

    let index_to_class = Arc::new(index_to_class);
    let model_config = Arc::new(ModelConfig::default());

    // Load the model from MODEL_PATH via a VarMap
    let mut varmap = VarMap::new();
    varmap.load(MODEL_PATH)?;

    let device = Device::cuda_if_available(0)?;

    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Arc::new(CategoriesPredictorModel::new(&vs, &model_config)?);

    // Build the shared data
    let shared_data = SharedData {
        word_to_index: Arc::clone(&word_to_index),
        index_to_class: Arc::clone(&index_to_class),
        model: Arc::clone(&model),
    };

    let health_check_route = warp::get()
        .and(warp::path("hc"))
        .map(|| warp::reply::json(&serde_json::json!({"status": "healthy"})));

    let predict_route = warp::post()
        .and(warp::path("predict"))
        .and(warp::body::json())
        .and(with_shared_data(shared_data))
        .and_then(|body: PredictRequest, data: SharedData| async move {
            match get_predictions(&body.text, &data.word_to_index, &data.model) {
                Ok(predictions) => {
                    let predicted_categories = map_to_class_names_with_scores(
                        predictions,
                        &data.index_to_class,
                        PREDICTION_THRESHOLD,
                    );
                    let response = PredictResponse {
                        predictions: predicted_categories,
                    };
                    Ok::<_, warp::Rejection>(warp::reply::json(&response))
                }
                Err(error) => Ok(warp::reply::json(
                    &serde_json::json!({"error": error.to_string()}),
                )),
            }
        });

    let routes = health_check_route.or(predict_route);

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;

    Ok(())
}
