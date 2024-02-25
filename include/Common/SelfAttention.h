// SelfAttention.h

#ifndef SELF_ATTENTION_H
#define SELF_ATTENTION_H

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include <numeric>
#include <stdexcept>
#include <random>
class SelfAttention {
public:
    explicit SelfAttention(int model_dim);

    std::vector<double> forward(const std::vector<double> &input) const;
    std::vector<double> multiply(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) const;
    // If you intend to use this class for both self-attention and encoder-decoder attention,
    // keep the encoder_input parameter; otherwise, remove it for a basic self-attention mechanism.

    std::vector<double> forward(const std::vector<double> &input, const std::vector<std::vector<double>> &encoder_input) const;

    std::vector<double> computeAttentionScores(const std::vector<double>& Q, const std::vector<std::vector<double>>& K);
private:
    int model_dim;
    std::vector<std::vector<double>> W_q, W_k, W_v;
    // Add other necessary member variables here, such as weights for the attention mechanism
};

#endif // SELF_ATTENTION_H
