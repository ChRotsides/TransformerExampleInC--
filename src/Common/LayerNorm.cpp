// LayerNorm.cpp
#include "../../include/Common/LayerNorm.h"

LayerNorm::LayerNorm(int model_dim) : model_dim(model_dim) { // Corrected: Scoped with the class name
    // In a real implementation, gamma and beta would be learnable parameters

    //TODO: initialize gamma and beta to random values

    gamma.resize(model_dim, 0.001f); // Initialized to ones
    beta.resize(model_dim, 0.0f);  // Initialized to zeros
}


std::vector<double> LayerNorm::normalize(const std::vector<double>& input) const {
    double mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();

    double variance = 0.0f;
    for (double val : input) {
        variance += (val - mean) * (val - mean);
    }
    variance /= input.size();

    const double epsilon = 1e-5; // Small constant to avoid division by zero
    std::vector<double> normalized(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        normalized[i] = (input[i] - mean) / std::sqrt(variance + epsilon);
        
        normalized[i] = gamma[0] * normalized[i] + beta[0];
    }

    return normalized;
}
