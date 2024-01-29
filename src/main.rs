mod common;
mod serving;

use std::sync::Arc;

use common::{
    create_class_mappings_from_class_names, create_vocabulary_to_index_mapping,
    make_mock_vocabulary, CategoriesPredictorModel, ModelConfig,
};
use serving::processing::{get_predictions, map_to_class_names_with_scores};
use serving::request_model::{PredictRequest, PredictResponse};
use warp::Filter;

#[tokio::main]
async fn main() {
    env_logger::init();

    let class_names: Vec<String> = vec!["sport".to_string(), "weather".to_string()];
    let vocabulary = make_mock_vocabulary();

    let mapping = Arc::new(create_vocabulary_to_index_mapping(&vocabulary));
    let (class_to_index, index_to_class) = create_class_mappings_from_class_names(class_names);

    let class_to_index = Arc::new(class_to_index);
    let index_to_class = Arc::new(index_to_class);

    let model_config = ModelConfig::default();

    let model = Arc::new(CategoriesPredictorModel::random(&model_config).unwrap());

    let health_check_route = warp::get()
        .and(warp::path("hc"))
        .map(|| warp::reply::json(&serde_json::json!({"status": "healthy"})));

    let predict_route = warp::post()
        .and(warp::path("predict"))
        .and(warp::body::json())
        .map({
            let mapping = Arc::clone(&mapping); // Clone Arc for the closure
            let class_to_index = Arc::clone(&class_to_index);
            let model = Arc::clone(&model);
            move |body: PredictRequest| match get_predictions(&body.text, &mapping, &model) {
                Ok(predictions) => {
                    let predicted_categories =
                        map_to_class_names_with_scores(predictions, &class_to_index, -1000.0);
                    let response = PredictResponse {
                        predictions: predicted_categories,
                    };
                    // Todo: There's a missing sigmoid
                    warp::reply::json(&response)
                }
                Err(error) => warp::reply::json(&serde_json::json!({"error": error.to_string()})),
            }
        });

    let routes = health_check_route.or(predict_route);

    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
