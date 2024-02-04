

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

mod config;
mod dataset;
mod transform;
mod metrics;

use metrics::f1_score;

use candle_nn::ops::sigmoid;
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam;
use candle_optimisers::adam::ParamsAdam;
use common::{
    create_class_mapping_from_labels, create_vocabulary_to_index_mapping, make_vocabulary, multi_hot_encode, store_index_to_class_mapping, store_vocabulary, MODEL_PATH, PREDICTION_THRESHOLD
};
use common::{CategoriesPredictorModel, ModelConfig};
use config::TrainConfig;
use dataset::{read_data, Dataset};
use transform::encode;

fn train(
    dataset: &Dataset,
    dev: &Device,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Result<()> {
    let train_data = dataset.train_data.to_device(dev)?;
    let train_labels = dataset.train_labels.to_device(dev)?;

    let test_data = dataset.test_data.to_device(dev)?;
    let test_labels = dataset.test_labels.to_device(dev)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = CategoriesPredictorModel::new(&vs, &model_config)?;

    let optimizer_params = adam::ParamsAdam {
        lr: train_config.learning_rate,
        ..ParamsAdam::default()
    };

    let mut optimizer = adam::Adam::new(varmap.all_vars(), optimizer_params)?;

    let n_epochs = train_config.n_epochs;

    let mut best_f1_score: f32 = 0.0;
    let mut best_model: VarMap = varmap.clone();

    let mut early_stopping_count: u8 = 0;

    for epoch in 1..n_epochs + 1 {
        let logits = model.forward(&train_data)?.flatten(0, 1)?;
        let loss = loss::binary_cross_entropy_with_logit(&logits, &train_labels)?;

        optimizer.backward_step(&loss)?;

        let test_logits = sigmoid(&model.forward(&test_data)?.flatten(0, 1)?)?;

        let test_predictions = test_logits
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .flat_map(|vec: &Vec<f32>| {
                vec.iter()
                    .map(|value: &f32| {
                        if value >= &PREDICTION_THRESHOLD {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<f32>>();
        let (n_samples, n_classes) = test_labels.dims2()?;

        let test_prediciton_tensor = Tensor::from_vec(test_predictions.clone(), (n_samples, n_classes), dev)?;
        
        log::info!("Test predictions {:?}", test_predictions);
        log::info!("Test labels: {:?}", test_labels.clone().to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<f32>>());

        let test_f1_score = f1_score(&test_prediciton_tensor, &test_labels)?;
        if test_f1_score > best_f1_score {
            early_stopping_count = 0;
            best_f1_score = test_f1_score;
            best_model = varmap.clone();
        }
        else {
            early_stopping_count += 1;
            if early_stopping_count == train_config.early_stop_patience {
                log::warn!("Early stopping triggered.");
                best_model.save(MODEL_PATH)?;
                return Ok(());
            }
        }

        log::info!(
            "Epoch: {epoch:3} Train loss: {:8.5} Test F1: {:5.2}%",
            loss.to_scalar::<f32>()?,
            test_f1_score
        );
    }

    best_model.save(MODEL_PATH)?;
    Ok(())
}

pub fn main() -> Result<()> {
    env_logger::init();

    let train_config = TrainConfig::default();
    let model_config = ModelConfig::default();

    let device = Device::cuda_if_available(0)?;

    // Load the data
    let (train_data, train_labels) = read_data("data/train.csv")?;
    let (test_data, test_labels) = read_data("data/test.csv")?;

    log::debug!("Train data sample: {:?}", train_data[0]);

    // Create the class to index mapping
    let (class_to_index, index_to_class) = create_class_mapping_from_labels(&train_labels);

    // Store the index to class mapping for inference
    store_index_to_class_mapping(&index_to_class, "data/index_to_class.json")?;

    log::debug!("Class to index {:?}", class_to_index);

    // Multi-hot encode the labels
    let train_labels_encoded = multi_hot_encode(train_labels, &class_to_index)?;
    let test_labels_encoded = multi_hot_encode(test_labels, &class_to_index)?;

    // Make the vocabulary and the vocabulary to index from the training data
    let vocabulary = make_vocabulary(&train_data);

    // Store the vocabulary to be loaded during inference
    store_vocabulary(&vocabulary, "data/vocab.json")?;

    let vocabulary_index_mapping = create_vocabulary_to_index_mapping(&vocabulary);

    let max_seq_len = model_config.max_seq_len;

    // Split string, convert to indices and pad to max length
    let train_data_tensor = encode(&train_data, max_seq_len, &vocabulary_index_mapping, &device)?;
    let test_data_tensor = encode(&test_data, max_seq_len, &vocabulary_index_mapping, &device)?;

    let n_classes = model_config.n_classes;

    // Reshape test labels to (n_samples, n_classes)
    let train_labels_tensor = Tensor::from_vec(
        train_labels_encoded.clone(),
        (train_labels_encoded.len() / n_classes, n_classes),
        &device,
    )?
    .to_dtype(DType::F32)?;

    // Reshape test labels to (n_samples, n_classes)
    let test_labels_tensor = Tensor::from_vec(
        test_labels_encoded.clone(),
        (test_labels_encoded.len() / n_classes, n_classes),
        &device,
    )?
    .to_dtype(DType::F32)?;

    log::debug!("Train data tensor {:?}", train_data_tensor);
    log::debug!("Train labels tensor: {:?}", train_labels_tensor);

    log::debug!("Test data tensor: {:?}", test_data_tensor);
    log::debug!("Test labels tensor: {:?}", test_labels_tensor);

    let dataset = Dataset {
        train_data: train_data_tensor,
        train_labels: train_labels_tensor,
        test_data: test_data_tensor,
        test_labels: test_labels_tensor,
    };

    log::info!("Started training.");
    train(&dataset, &device, model_config, train_config)?;

    Ok(())
}
