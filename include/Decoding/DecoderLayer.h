// DecoderLayer.h

#ifndef DECODERLAYER_H
#define DECODERLAYER_H

#include <vector>
#include "../Common/SelfAttention.h"
#include "../Common/FeedForwardNetwork.h"
#include "../Common/LayerNorm.h"
class DecoderLayer {
public:
    DecoderLayer(int model_dim) : selfAttention(model_dim), encDecAttention(model_dim), feedForward(model_dim), layerNorm1(model_dim), layerNorm2(model_dim), layerNorm3(model_dim) {}

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& decoderInput, const std::vector<std::vector<double>>& encoderOutput);
    std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& dL_dOutputs, const std::vector<std::vector<double>>& encoderOutputs);
private:
    SelfAttention selfAttention;
    SelfAttention encDecAttention;  // For encoder-decoder attention
    FeedForwardNetwork feedForward;
    LayerNorm layerNorm1, layerNorm2, layerNorm3;

    std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) const {
        std::vector<double> result(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            result[i] = a[i] + b[i];
        }
        return result;
    }
};

#endif // DECODERLAYER_H