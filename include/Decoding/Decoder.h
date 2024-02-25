// Decoder.h
#ifndef DECODER_H
#define DECODER_H
#include <vector>
#include "DecoderLayer.h"

class Decoder
{
public:
    Decoder(int num_layers, int model_dim);

    std::vector<std::vector<double>> *forward(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> &encoderOutput);
    void backward(const std::vector<std::vector<double>> &dL_dDecoderOutputs, const std::vector<std::vector<double>> &encoderOutputs);

private:
    int num_layers, model_dim;
    std::vector<DecoderLayer> layers;
};

#endif // DECODER_H