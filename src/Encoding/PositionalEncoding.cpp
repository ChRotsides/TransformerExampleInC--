// PositionalEncoding.cpp
#include "../../include/Encoding/PositionalEncoding.h"
PositionalEncoding::PositionalEncoding(int max_seq_len, int model_dim) : max_seq_len(max_seq_len), model_dim(model_dim){
    encoding_matrix.resize(max_seq_len, std::vector<double>(model_dim));
    for (int pos = 0; pos < max_seq_len; ++pos)
    {
        for (int i = 0; i < model_dim; i += 2)
        {
            double position = pos / std::pow(10000.0, 2.0 * i / model_dim);
            encoding_matrix[pos][i] = std::sin(position);
            if (i + 1 < model_dim)
            {
                encoding_matrix[pos][i + 1] = std::cos(position);
            }
        }
    }
}

const std::vector<double>& PositionalEncoding::getEncoding(int pos){
    if (pos < 0 || pos >= max_seq_len)
    {
        throw std::out_of_range("PositionalEncoding::getEncoding: Position out of range.");
    }
    return encoding_matrix[pos];
}

std::vector<std::vector<double>> *PositionalEncoding::getEncodings(std::vector<std::vector<double>> &embeddings){
    int seq_len = std::min(embeddings.size(), encoding_matrix.size());
    // int seq_len = this->max_seq_len;
    // std::cout<<seq_len<<std::endl;
    // std::cout<<model_dim<<std::endl;
    // std::cout<<embeddings.size()<<std::endl;
    // std::cout<<embeddings[0].size()<<std::endl;
    std::vector<std::vector<double>> *encodings = new std::vector<std::vector<double>>(seq_len, std::vector<double>(this->model_dim));
    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < model_dim; ++j)
        {
            (*encodings)[i][j] = embeddings[i][j] + encoding_matrix[i][j];
        }
    }
    return encodings;
}