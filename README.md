# Headline Classifier

A simple and powerful end-to-end short text classification project in Rust, using Polars, Candle and Warp.

# Features

- 🧘 Simple NLP neural network training using 🤗 [Candle](https://github.com/huggingface/candle)
- 🚀 CSV I/O and processing with [Polars](https://pola.rs/)
- ⚡️ Inference via HTTP with [Warp](https://github.com/seanmonstar/warp)

# Usage

## Training

To train a model, simply run the inference

```bash
RUST_LOG=info cargo run --bin training
```

## Inference

To start the prediction service over HTTP, run:

```bash
RUST_LOG=warn cargo run --bin inference
```

Then, to get the predictions, send a request via CURL/Postman:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "blizzard hits southwest"}' http://localhost:3030/predict
```