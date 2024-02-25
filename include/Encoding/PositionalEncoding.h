// PositionalEncoding.h
#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
class PositionalEncoding {
public:
    PositionalEncoding(int max_seq_len, int model_dim);
    const std::vector<double>& getEncoding(int pos);
    std::vector<std::vector<double>> *getEncodings(std::vector<std::vector<double>> &embeddings);

private:
    int max_seq_len;
    int model_dim;
    std::vector<std::vector<double>> encoding_matrix;
};

#endif // POSITIONAL_ENCODING_H