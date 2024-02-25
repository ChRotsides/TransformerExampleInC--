// DecoderLayer.cpp
#include "../../include/Decoding/DecoderLayer.h"

 std::vector<std::vector<double>> DecoderLayer::forward(const std::vector<std::vector<double>>& decoderInput, const std::vector<std::vector<double>>& encoderOutput) {
        // Self-attention on the decoder input
        std::vector<std::vector<double>> selfAttnOutput(decoderInput.size(), std::vector<double>(decoderInput[0].size()));
        for (size_t i = 0; i < decoderInput.size(); ++i) {
            selfAttnOutput[i] = selfAttention.forward(decoderInput[i]);
        }

        // Add & Norm after self-attention
        std::vector<std::vector<double>> addNorm1(decoderInput.size(), std::vector<double>(decoderInput[0].size()));
        for (size_t i = 0; i < decoderInput.size(); ++i) {
            addNorm1[i] = layerNorm1.normalize(add(decoderInput[i], selfAttnOutput[i]));
        }

        // Encoder-decoder attention using the encoder output
        std::vector<std::vector<double>> encDecAttnOutput(addNorm1.size(), std::vector<double>(addNorm1[0].size()));
        for (size_t i = 0; i < addNorm1.size(); ++i) {
            encDecAttnOutput[i] = encDecAttention.forward(addNorm1[i], encoderOutput);  // Assuming the encDecAttention.forward method is designed to handle this case
        }

        // Add & Norm after encoder-decoder attention
        std::vector<std::vector<double>> addNorm2(addNorm1.size(), std::vector<double>(addNorm1[0].size()));
        for (size_t i = 0; i < addNorm1.size(); ++i) {
            addNorm2[i] = layerNorm2.normalize(add(addNorm1[i], encDecAttnOutput[i]));
        }

        // Feed-forward network
        std::vector<std::vector<double>> ffOutput(addNorm2.size(), std::vector<double>(addNorm2[0].size()));
        for (size_t i = 0; i < addNorm2.size(); ++i) {
            ffOutput[i] = feedForward.forward(addNorm2[i]);
        }

        // Final Add & Norm
        std::vector<std::vector<double>> addNorm3(addNorm2.size(), std::vector<double>(addNorm2[0].size()));
        for (size_t i = 0; i < addNorm2.size(); ++i) {
            addNorm3[i] = layerNorm3.normalize(add(addNorm2[i], ffOutput[i]));
        }

        return addNorm3;
    }
        std::vector<std::vector<double>> DecoderLayer::backward(const std::vector<std::vector<double>>& dL_dOutputs, const std::vector<std::vector<double>>& encoderOutputs) {
        // Compute gradients with respect to layer inputs and parameters here
        // Return gradients with respect to layer inputs
        return dL_dOutputs;
        }