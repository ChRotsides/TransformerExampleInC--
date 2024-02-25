// Decoder.cpp

#include "../../include/Decoding/Decoder.h"

Decoder::Decoder(int num_layers, int model_dim) : num_layers(num_layers), model_dim(model_dim) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(DecoderLayer(model_dim));
    }
}

std::vector<std::vector<double>> *Decoder::forward(const std::vector<std::vector<double>>& input, const std::vector<std::vector<double>>& encoderOutput) {
    std::vector<std::vector<double>> *outputs = new std::vector<std::vector<double>>(input);
    for (auto& layer : layers) {
        (*outputs) = layer.forward((*outputs), encoderOutput);
    }
    return outputs;
}

void Decoder::backward(const std::vector<std::vector<double>>& dL_dDecoderOutputs, const std::vector<std::vector<double>>& encoderOutputs) {
    std::vector<std::vector<double>> dL_dInputs = dL_dDecoderOutputs;

    // Iterate through layers in reverse order for backpropagation
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        dL_dInputs = it->backward(dL_dInputs, encoderOutputs);
    }
}
