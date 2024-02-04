use candle_core::{Result, Tensor};

use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

use candle_core::Device;

pub struct ModelConfig {
    pub device: Device,
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub n_classes: usize,
    pub max_seq_len: usize,
}

pub const MAX_SEQ_LEN: usize = 128;

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            vocab_size: 1000,
            embedding_size: 15,
            hidden_size: 20,
            n_classes: 2,
            max_seq_len: MAX_SEQ_LEN,
        }
    }
}

#[derive(Debug)]
pub struct HeadlineClassifierModel {
    embedding: Embedding,
    fully_connected: Linear,
    classifier: Linear,
}

impl HeadlineClassifierModel {
    /// Create a new instance of a model using the provided VarBuilder and ModelConfig.
    ///
    /// # Arguments
    ///
    /// * `vb` - A reference to a VarBuilder used for creating variables.
    /// * `config` - A reference to a ModelConfig containing configuration parameters for the model.
    ///
    /// # Errors
    ///
    /// This function can return an error if there are issues with creating embedding or linear layers using the VarBuilder, or if any other initialization fails.
    ///
    /// # Returns
    ///
    /// This function returns a `Result<Self>`, where `Self` represents the newly created model instance on success.
    pub fn new(vb: &VarBuilder, config: &ModelConfig) -> Result<Self> {
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
