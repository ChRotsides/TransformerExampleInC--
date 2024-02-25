// FinalLayer.h
#ifndef FINAL_LAYER_H
#define FINAL_LAYER_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

class FinalLayer {
public:
    explicit FinalLayer(int input_dim, int output_dim);

    std::vector<double> forward(const std::vector<double>& input) const;
    void updateParameters(std::vector<double>& gradients,int index, double learning_rate);

private:
    int input_dim, output_dim;
    std::vector<std::vector<double>> W; // Weights for the linear layer
    std::vector<double> b; // Bias for the linear layer

    std::vector<double> linear(const std::vector<double>& input) const;
    std::vector<double> softmax(const std::vector<double>& input) const;
};

#endif // FINAL_LAYER_H
