// FeedForwardNetwork.h

#ifndef FEED_FORWARD_NETWORK_H
#define FEED_FORWARD_NETWORK_H

#include <vector>
#include <iostream>
#include <cmath>

class FeedForwardNetwork {
public:
    explicit FeedForwardNetwork(int model_dim);

    std::vector<double> forward(const std::vector<double>& input) const;

private:
    int model_dim;
    // In a real implementation, you would have weights and biases here
};


#endif // FEED_FORWARD_NETWORK_H