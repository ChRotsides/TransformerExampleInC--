// static class for utility functions utils.h

// utils.h

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

class Utils {
public:
    static bool containsNaN(const std::vector<double>& vec) {
        return std::any_of(vec.begin(), vec.end(), [](float x) { return std::isnan(x); });
    }
    static std::vector<double> softmax(const std::vector<double>& scores) {
        std::vector<double> expScores(scores.size());
        double sumExpScores = 0.0;

        // Compute the exponentiated scores and their sum
        for (size_t i = 0; i < scores.size(); ++i) {
            expScores[i] = std::exp(scores[i]);
            sumExpScores += expScores[i];
        }

        // Normalize the exponentiated scores by their sum
        for (size_t i = 0; i < expScores.size(); ++i) {
            expScores[i] /= sumExpScores;
        }

    return expScores;
}

};

#endif // UTILS_H