pub struct TrainConfig {
    pub n_epochs: u32,
    pub learning_rate: f64,
    pub early_stop_patience: u8
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            n_epochs: 100,
            learning_rate: 0.001,
            early_stop_patience: 20
        }
    }
}
