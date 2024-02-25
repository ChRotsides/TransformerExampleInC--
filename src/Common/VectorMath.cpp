// VectorMath.cpp

#include "../../include/Common/VectorMath.h"

// Function to transpose a matrix
std::vector<std::vector<double>> VectorMath::transpose(const std::vector<std::vector<double>>& matrix) {
    std::vector<std::vector<double>> result(matrix[0].size(), std::vector<double>(matrix.size()));
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b[0].size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b[0].size(); ++j) {
            for (size_t k = 0; k < b.size(); ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}