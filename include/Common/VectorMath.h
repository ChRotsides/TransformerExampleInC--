#ifndef VECTOR_MATH_H    // Include guard
#define VECTOR_MATH_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
class VectorMath {
public:
    // Function to perform matrix multiplication
    static std::vector<std::vector<double>> matmul(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) ;

    // Function to transpose a matrix
    static std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix);



};

#endif // VECTOR_MATH_H
