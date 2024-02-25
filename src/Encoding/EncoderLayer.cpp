// EncoderLayer.cpp
#include "../../include/Encoding/EncoderLayer.h"

EncoderLayer::EncoderLayer(int model_dim) : selfAttention(model_dim), feedForward(model_dim), layerNorm(model_dim) {}
bool containsNaN(const std::vector<double>& vec) {
    return std::any_of(vec.begin(), vec.end(), [](float x) { return std::isnan(x); });
}


std::vector<std::vector<double>> EncoderLayer::forward(const std::vector<std::vector<double>>& inputs) {
    std::vector<std::vector<double>> attentionOutputs;
    // Apply self-attention to each input in the sequence


    for (const auto& input : inputs) {
        auto attentionOutput = selfAttention.forward(input); // Assuming selfAttention can handle a single embedding
        if (containsNaN(attentionOutput)) {
            for (auto x : attentionOutput) {
                std::cerr << x << " ";
            }
            std::cerr << std::endl;
            for (auto x : input) {
                std::cerr << x << " ";
            }
            
            std::cerr << "NaN detected after self-attention" << std::endl;
            throw std::runtime_error("NaN detected after self-attention");  
        }
        attentionOutputs.push_back(attentionOutput);
            // Use this function to check for NaNs after major operations, e.g.,
    }

    std::vector<std::vector<double>> addNorm1Outputs;
    // Apply Add & Norm to the result of self-attention
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto addNorm1 = layerNorm.normalize(add(inputs[i], attentionOutputs[i])); // Assuming layerNorm can handle a single embedding
                if (containsNaN(addNorm1)) {
            std::cerr << "NaN detected after addNorm1" << std::endl;
            throw std::runtime_error("NaN detected after addNorm1");
        }
        addNorm1Outputs.push_back(addNorm1);
    }

    std::vector<std::vector<double>> ffOutputs;
    // Apply feed-forward network to each result of Add & Norm
    for (const auto& addNorm1Output : addNorm1Outputs) {
        auto ffOutput = feedForward.forward(addNorm1Output); // Assuming feedForward can handle a single embedding
                        if (containsNaN(ffOutput)) {
            std::cerr << "NaN detected after ffOutput" << std::endl;
            throw std::runtime_error("NaN detected after ffOutput");
        }
        ffOutputs.push_back(ffOutput);
    }

    std::vector<std::vector<double>> addNorm2Outputs;
    // Apply the second Add & Norm
    for (size_t i = 0; i < addNorm1Outputs.size(); ++i) {
        auto addNorm2 = layerNorm.normalize(add(addNorm1Outputs[i], ffOutputs[i])); // Assuming layerNorm can handle a single embedding
                                if (containsNaN(addNorm2)) {
            std::cerr << "NaN detected after addNorm2" << std::endl;
            throw std::runtime_error("NaN detected after addNorm2");
        }
        addNorm2Outputs.push_back(addNorm2);
    }

    return addNorm2Outputs;
}


std::vector<double> EncoderLayer::add(const std::vector<double>& a, const std::vector<double>& b) const {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}