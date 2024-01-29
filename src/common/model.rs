use candle_core::{Result, Tensor};
use candle_nn::ops::sigmoid;
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

use candle_core::Device;
use candle_optimisers::Model;

pub struct ModelConfig {
    pub device: Device,
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub n_classes: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            vocab_size: 1000,
            embedding_size: 100,
            hidden_size: 30,
            n_classes: 2,
        }
    }
}

#[derive(Debug)]
pub struct CategoriesPredictorModel {
    embedding: Embedding,
    fully_connected: Linear,
    classifier: Linear,
}

impl CategoriesPredictorModel {
    pub fn random(config: &ModelConfig) -> Result<Self> {
        let embedding = Embedding::new(
            Tensor::rand(
                -0.1f32,
                0.1f32,
                &[config.vocab_size, config.embedding_size],
                &config.device,
            )?,
            config.embedding_size,
        );
        let fully_connected = Linear::new(
            Tensor::rand(
                -0.1f32,
                0.1f32,
                &[config.hidden_size, config.embedding_size],
                &config.device,
            )?,
            Some(Tensor::rand(
                -0.1f32,
                0.1f32,
                &[config.hidden_size],
                &config.device,
            )?),
        );
        let classifier = Linear::new(
            Tensor::rand(
                -0.1f32,
                0.1f32,
                &[config.n_classes, config.hidden_size],
                &config.device,
            )?,
            Some(Tensor::rand(
                -0.1f32,
                0.1f32,
                &[config.n_classes],
                &config.device,
            )?),
        );

        Ok(Self {
            embedding,
            fully_connected,
            classifier,
        })
    }

    pub fn new(vb: VarBuilder, config: ModelConfig) -> Result<Self> {
        let embedding = embedding(config.vocab_size, config.embedding_size, vb.pp("embedding"))?;
        let fully_connected = linear(config.embedding_size, config.hidden_size, vb.pp("linear"))?;
        let classifier = linear(config.hidden_size, config.n_classes, vb.pp("classifier"))?;
        Ok(Self {
            embedding,
            fully_connected,
            classifier,
        })
    }

    pub fn forward(&self, input_indices: &Tensor) -> Result<Tensor> {
        let embeddings = self.embedding.forward(input_indices)?;
        log::debug!(
            "Embeddings - Shape: {:?}. Values: {:?}",
            embeddings.shape(),
            embeddings.get(0).unwrap().get(0)
        );

        let mean_embedding = embeddings.mean_keepdim(0)?;
        log::debug!(
            "Mean embedding - Shape {:?}. Values {:?}",
            mean_embedding.shape(),
            mean_embedding.get(0)
        );

        let features = self.fully_connected.forward(&mean_embedding)?;
        log::debug!(
            "Features - Shape: {:?}, Values {:?}",
            features.shape(),
            features.get(0)
        );
        self.classifier.forward(&features)
    }
}