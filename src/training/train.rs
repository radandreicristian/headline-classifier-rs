// use candle_core::{Device};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

mod config;
mod dataset;

use candle_nn::ops::sigmoid;
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use candle_optimisers::adam;
use candle_optimisers::adam::ParamsAdam;
use common::{
    create_class_mapping_from_labels, create_vocabulary_to_index_mapping, make_vocabulary,
    map_words_to_indices, multi_hot_encode, pad_vector,
};
use common::{CategoriesPredictorModel, ModelConfig};
use config::TrainConfig;
use dataset::Dataset;

fn train(
    dataset: Dataset,
    dev: &Device,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> anyhow::Result<CategoriesPredictorModel> {
    let train_data = dataset.train_data.to_device(dev)?;
    let train_labels = dataset.train_labels.to_device(dev)?;

    let test_data = dataset.test_data.to_device(dev)?;
    let test_labels = dataset.test_labels.to_device(dev)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = CategoriesPredictorModel::new(vs, model_config)?;

    let optimizer_params = adam::ParamsAdam {
        lr: train_config.learning_rate,
        ..ParamsAdam::default()
    };
    let mut optimizer = adam::Adam::new(varmap.all_vars(), optimizer_params)?;

    let n_epochs = train_config.n_epochs;
    let mut final_accuracy: f32 = 0.0;
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
                    .map(|value: &f32| if value >= &0.5f32 { 1.0 } else { 0.0 })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<f32>>();

        let test_prediciton_tensor = Tensor::from_vec(test_predictions, (4, 2), dev)?;

        let sum_ok = test_prediciton_tensor
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let test_accuracy = sum_ok / 8.;

        final_accuracy = 100. * test_accuracy;

        println!(
            "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
            loss.to_scalar::<f32>()?,
            final_accuracy
        );
        if final_accuracy == 100.0 {
            break;
        }
    }
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        Ok(model)
    }
}

pub fn main() -> Result<()> {
    env_logger::init();

    let device = Device::cuda_if_available(0)?;

    // Load or hard-code the train data -- list of strings
    // Todo - See if there's a workaround for this.
    let train_data: Vec<String> = [
        "football player leads team to victory in overtime win",
        "tennis player dominates competition in surprise comeback",
        "blizzard warning looms over midwest",
        "heatwave sweeps across south",
        "groundbreaking medical discovery promises hope for illness",
        "tech giant unveils revolutionary new device",
        "celebrity couple's surpsie wedding leaves fans in awe",
        "football game interrupted by blizzard",
    ]
    .iter()
    .map(|&sentence| sentence.to_string())
    .collect();

    let train_data_refs: Vec<&str> = vec![
        "player leads team to victory in overtime win",
        "tennis player dominates competition in surprise comeback",
        "blizzard warning looms over midwest",
        "heatwave sweeps across south",
        "groundbreaking medical discovery promises hope for illness",
        "tech giant unveils revolutionary new device",
        "celebrity couple's surpsie wedding leaves fans in awe",
        "history exhibistion showcase masterpieces from around the world",
    ];

    let test_data: Vec<String> = [
        "injury strikes star player of football team in overtime",
        "no risk of blizzard or heatwave today!",
        "tech leaders convene to address AI crisis",
        "local communitiy rallies together to aid neighbours affected by disaster",
    ]
    .iter()
    .map(|&sentence| sentence.to_string())
    .collect();

    // It's actually (2, 6) flattened
    // let train_labels: Vec<u32> = vec![1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
    let train_labels: Vec<&str> = vec![
        "sports",
        "sports",
        "weather",
        "weather",
        "",
        "",
        "",
        "sports,weather",
    ];

    // It's actually (2, 4) flattened
    //let test_labels: Vec<u32> = vec![1, 0, 0, 1, 0, 0, 0, 0];
    let test_labels: Vec<&str> = vec!["sports", "weather", "", ""];

    let (class_to_index, index_to_class) = create_class_mapping_from_labels(&train_labels);

    log::error!("Class to index {:?}", class_to_index);

    let train_labels_encoded = multi_hot_encode(train_labels, &class_to_index);
    let test_labels_encoded = multi_hot_encode(test_labels, &class_to_index);

    // Make vocabulary and string to index mapping
    let vocabulary = make_vocabulary(train_data_refs);

    let vocabulary_index_mapping = create_vocabulary_to_index_mapping(vocabulary);

    // Split string, convert to indices and pad to max length
    let train_indices: Vec<u32> = train_data
        .iter()
        .flat_map(|sentence| {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let indices = map_words_to_indices(words, &vocabulary_index_mapping);
            pad_vector(indices, 128, 0)
        })
        .collect();

    let train_data_tensor = Tensor::from_vec(
        train_indices.clone(),
        (128, train_indices.len() / 128),
        &device,
    )?;

    let test_indices: Vec<u32> = test_data
        .iter()
        .flat_map(|sentence| {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let indices = map_words_to_indices(words, &vocabulary_index_mapping);
            pad_vector(indices, 128, 0)
        })
        .collect();

    let test_data_tensor = Tensor::from_vec(
        test_indices.clone(),
        (128, test_indices.len() / 128),
        &device,
    )?;

    let train_labels_tensor = Tensor::from_vec(
        train_labels_encoded.clone(),
        (train_labels_encoded.len() / 2, 2),
        &device,
    )?
    .to_dtype(DType::F32)?;

    let test_labels_tensor = Tensor::from_vec(
        test_labels_encoded.clone(),
        (test_labels_encoded.len() / 2, 2),
        &device,
    )?
    .to_dtype(DType::F32)?;

    println!("Train data tensor {:?}", train_data_tensor);
    println!("Train labels tensor: {:?}", train_labels_tensor);

    println!("Test data tensor: {:?}", test_data_tensor);
    println!("Test labels tensor: {:?}", test_labels_tensor);

    let dataset = Dataset {
        train_data: train_data_tensor,
        train_labels: train_labels_tensor,
        test_data: test_data_tensor,
        test_labels: test_labels_tensor,
    };

    let trained_model: CategoriesPredictorModel;

    let train_config = TrainConfig::default();
    let model_config = ModelConfig::default();

    println!("Trying to train a classifier.");
    match train(dataset.clone(), &device, model_config, train_config) {
        Ok(model) => {
            trained_model = model;
            log::info!("Model {:?}", trained_model);
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }

    Ok(())
}
