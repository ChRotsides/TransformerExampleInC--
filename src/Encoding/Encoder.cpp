// Encoder.cpp
#include "../../include/Encoding/Encoder.h"
Encoder::Encoder(int num_layers, int model_dim) : num_layers(num_layers), model_dim(model_dim) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(EncoderLayer(model_dim));
    }
}

std::vector<std::vector<double>> Encoder::forward(const std::vector<std::vector<double>>& inputs) {
    auto outputs = inputs;
    for (auto& layer : this->layers) {
        // Each layer processes the entire sequence of embeddings
        outputs = layer.forward(outputs);
    }
    return outputs;
}