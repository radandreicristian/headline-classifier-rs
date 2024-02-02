use common;
mod inference;
mod types;

use std::collections::HashMap;
use std::sync::Arc;

use common::{
    load_index_to_class_mapping, create_vocabulary_to_index_mapping,
    load_vocabulary, CategoriesPredictorModel, ModelConfig,
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
    model: Arc<CategoriesPredictorModel>
}

#[tokio::main]
async fn main() -> anyhow::Result<()>{
    
    env_logger::init();

    let vocabulary = load_vocabulary("data/vocab.json")?;

    let word_to_index = Arc::new(create_vocabulary_to_index_mapping(&vocabulary));
    let index_to_class = load_index_to_class_mapping("data/index_to_class.json")?;

    let index_to_class = Arc::new(index_to_class);

    let model_config = Arc::new(ModelConfig::default());

    let model = Arc::new(CategoriesPredictorModel::random(&Arc::clone(&model_config))?);

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
                    let predicted_categories =
                        map_to_class_names_with_scores(predictions, &data.index_to_class, 0.3);
                    let response = PredictResponse {
                        predictions: predicted_categories,
                    };
                    // Todo: There's a missing sigmoid
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
