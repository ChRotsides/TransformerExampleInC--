// Encoder.h

#ifndef ENCODER_H
#define ENCODER_H

#include <vector>

#include "EncoderLayer.h"


class Encoder {
public:
    Encoder(int num_layers, int model_dim);

    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& inputs);
private:
    int num_layers, model_dim;
    std::vector<EncoderLayer> layers;
};

#endif // ENCODER_H