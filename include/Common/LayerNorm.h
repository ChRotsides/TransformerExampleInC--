// LayerNorm.h

#ifndef LAYERNORM_H
#define LAYERNORM_H


#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

class LayerNorm {
public:
    explicit LayerNorm(int model_dim); // Corrected: Removed inline initialization and added semicolon

    std::vector<double> normalize(const std::vector<double>& input) const; // Added 'const' to match the definition
private:
    int model_dim;
    std::vector<double> gamma, beta;
};

#endif // LAYERNORM_H