// FinalLayer.cpp
#include "../../include/Common/FinalLayer.h"

FinalLayer::FinalLayer(int input_dim, int output_dim) : input_dim(input_dim), output_dim(output_dim) {
    W.resize(output_dim, std::vector<double>(input_dim, 0.1f)); // Placeholder initialization
    b.resize(output_dim, 0.0f); // Placeholder initialization
}

std::vector<double> FinalLayer::forward(const std::vector<double>& input) const {
    auto z = linear(input);
    return softmax(z);
}

std::vector<double> FinalLayer::linear(const std::vector<double>& input) const {
    std::vector<double> output(output_dim, 0.0f);
    for (int i = 0; i < output_dim; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            output[i] += W[i][j] * input[j];
        }
        output[i] += b[i];
    }
    return output;
}
void FinalLayer::updateParameters(std::vector<double>& gradients,int index, double learning_rate) {
    // Assuming weights and gradients are 2D vectors of the same shape

        for (size_t j = 0; j < W[index].size(); ++j) {
            W[index][j] -= learning_rate * gradients[j];  // Update rule for SGD
        }

}
std::vector<double> FinalLayer::softmax(const std::vector<double>& input) const {
    std::vector<double> output(input.size(), 0.0f);
    double maxElement = *max_element(input.begin(), input.end());
    double sum = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - maxElement); // Improve numerical stability
        sum += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum;
    }

    return output;
}
