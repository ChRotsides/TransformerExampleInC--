// EncoderLayer.h
#ifndef ENCODERLAYER_H
#define ENCODERLAYER_H

#include <vector>
#include "../Common/SelfAttention.h"
#include "../Common/FeedForwardNetwork.h"
#include "../Common/LayerNorm.h"
#include <iostream>
#include <cmath>


class EncoderLayer {
public:
    EncoderLayer(int model_dim);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs) ;

private:
    SelfAttention selfAttention;
    FeedForwardNetwork feedForward;
    LayerNorm layerNorm;

    std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) const;
};

#endif // ENCODERLAYER_H