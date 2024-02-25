// main.cpp
#include <iostream>
#include <fstream>
#include "./include/Common/Embedding.h"
#include "./include/Common/Tokenizer.h"
#include "./include/Encoding/PositionalEncoding.h"
#include "./include/Encoding/Encoder.h"
#include "./include/Decoding/Decoder.h"
#include "./include/Common/FinalLayer.h"
#include "./include/Common/SelfAttention.h"
#include "./include/Common/VectorMath.h"

double computeCrossEntropyLoss(const std::vector<double> &predictedProbabilities, int targetTokenID)
{
    // Get the predicted probability for the target token
    double predictedProbability = predictedProbabilities[targetTokenID];

    // Calculate the cross-entropy loss
    double epsilon = 1e-8; // Small constant to avoid log(0)
    double loss = -log(predictedProbability + epsilon);

    return loss;
}
std::vector<double> computeGradientOfLossWrtLayerOutput(const std::vector<double>& predictions, const std::vector<double>& trueLabels) {
    std::vector<double> gradients(predictions.size());

    for (size_t i = 0; i < predictions.size(); ++i) {
        gradients[i] = predictions[i] - trueLabels[i];  // Gradient for cross-entropy loss
    }

    return gradients;
}

int main()
{
    Tokenizer tok;
    std::fstream file;

    file.open("./saves/text_test.txt", std::ios::in);
    std::vector<std::string> text;

    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty())
            {
                continue;
            }
            text.push_back(line);
        }
        file.close();
    }
    else
    {
        std::cout << "Error: File not found" << std::endl;
        return 1;
    }

    std::cout << text.size() << std::endl;

    std::vector<std::string> input_text;
    std::vector<std::string> output_text;
    std::vector<std::vector<int>> input_tokens;
    std::vector<std::vector<int>> output_tokens;
    for (int i = 0; i < text.size(); i += 2)
    {
        input_text.push_back(text[i]);
        output_text.push_back(text[i + 1]);
        input_tokens.push_back(tok.tokenize(text[i]));
        output_tokens.push_back(tok.tokenize(text[i + 1]));
    }
    std::string end_token = "<end>";
    int end_token_id = tok.tokenize(end_token)[0];
    std::vector<std::unordered_map<int, double>> losses = std::vector<std::unordered_map<int, double>>();
    std::vector<std::vector<double>> gradients;
    for (int i = 0; i < input_text.size(); i++)
    {

        int model_dim = 128;
        int vocab_size = tok.getVocabSize();
        Embedding embedding(vocab_size, model_dim);

        std::vector<std::vector<double>> *embedded_input = embedding.tokenToEmbeddings(input_tokens[i]);
        std::vector<std::vector<double>> *embedded_output = embedding.tokenToEmbeddings(output_tokens[i]);
        PositionalEncoding pe(640, model_dim);
        std::vector<std::vector<double>> *encoded_inputs = pe.getEncodings(*embedded_input);
        std::vector<std::vector<double>> *encoded_outputs = pe.getEncodings(*embedded_output);

        Encoder encoder(6, model_dim);

        std::vector<std::vector<double>> encoder_outputs = encoder.forward(*encoded_inputs);

        Decoder decoder(6, model_dim);

        std::vector<std::vector<double>> *decoder_outputs = decoder.forward((*encoded_inputs), encoder_outputs);

        // Initialize the Final Linear and Softmax Layer
        FinalLayer finalLayer(model_dim, vocab_size);
        std::vector<std::vector<double>> output_probabilities;

        // Process each output from the decoder through the final layer
        for (const auto &decoder_output : *decoder_outputs)
        {
            // Get probabilities for each token in the sequence
            std::vector<double> token_probabilities = finalLayer.forward(decoder_output);
            output_probabilities.push_back(token_probabilities);
        }

        // Output the result (for illustration, print the probabilities of the first token)
        // std::cout << "Output probabilities for each token in the sequence:" << std::endl;

        // Iterate over each set of output probabilities in the sequence
        std::vector<int> resp_tokens;
        std::vector<int> resp_tokens_max;
        std::default_random_engine generator;
        // std::cout << "Size Output Probabilites: " << output_probabilities.size() << std::endl;
        // std::cout << "Size Output Tokens: " << output_tokens[i].size() << std::endl;

        size_t size_difference = std::max(output_probabilities[i].size(), output_tokens[i].size()) - std::min(output_probabilities[i].size(), output_tokens[i].size());

        if (output_probabilities[i].size() < output_tokens[i].size())
        {
            // Pad output_probabilities with values that sum up to 1, last being 0.51
            double remaining_probability = 0.49;                              // Total remaining probability to distribute
            double pad_value = remaining_probability / (size_difference - 1); // Distribute evenly among all but the last
            output_probabilities[i].insert(output_probabilities[i].end(), size_difference - 1, pad_value);
            output_probabilities[i].push_back(0.51); // Last probability is 0.51 for end_token_id
        }
        else if (output_probabilities[i].size() > output_tokens[i].size())
        {
            // Pad output_tokens with end_token_id
            output_tokens[i].insert(output_tokens[i].end(), size_difference, end_token_id);

            // Adjust output_probabilities to sum up to 1, with last being 0.51
            output_probabilities[i].back() = 0.51; // Set the last probability to 0.51 for end_token_id
            double sum = std::accumulate(output_probabilities[i].begin(), output_probabilities[i].end() - 1, 0.0);
            double remaining_probability = 0.49; // Remaining probability to distribute
            std::transform(output_probabilities[i].begin(), output_probabilities[i].end() - 1, output_probabilities[i].begin(),
                           [sum, remaining_probability](double &c)
                           { return (c / sum) * remaining_probability; });
        }

        // std::cout << "After Padding" << std::endl;
        // std::cout << "Size Output Probabilites: " << output_probabilities.size() << std::endl;
        // std::cout << "Size Output Tokens: " << output_tokens[i].size() << std::endl;
        for (int j = 0; j < output_tokens[i].size(); j++)
        {

            std::vector<double> output_probability = output_probabilities[i];
            int target_token_id = output_tokens[i][j];
            std::discrete_distribution<int> distribution(output_probability.begin(), output_probability.end());
            int sampled_token_id = distribution(generator);
            // std::cout << "Sampled Token ID: " << sampled_token_id << std::endl;

            // Output the sampled token ID
            // std::cout << "Sampled Token ID: " << sampled_token_id << std::endl;
            resp_tokens.push_back(sampled_token_id);
            // get the max
            int max_token_id = std::distance(output_probability.begin(), std::max_element(output_probability.begin(), output_probability.end()));
            double max_propability = output_probability[max_token_id];
            // std::cout << "Max Token ID: " << max_token_id<< "Propability: " << max_propability  << std::endl;
            // std::cout << "Target Token ID: " << target_token_id << std::endl;
            resp_tokens_max.push_back(max_token_id);

            double loss = computeCrossEntropyLoss(output_probability, target_token_id);

            // Output the loss
            // std::cout << "Cross-entropy loss: " << loss << std::endl;
            std::unordered_map<int, double> temp_loss_map;
            temp_loss_map[target_token_id] = loss;
            losses.push_back(temp_loss_map);

            // std::vector<double> temp_gradient = computeGradientOfLossWrtLayerOutput(output_probability, tok.oneHotEncode(target_token_id)[0]);

            // for (size_t k = 0; k < gradients.size(); ++k) {
            //         temp_gradient[k] = output_probability[k] > 0 ? temp_gradient[k]:0;
            //         finalLayer.updateParameters(temp_gradient,k, 0.01);
            //     }
            // }
            
        }

        


        std::string response = tok.detokenize(resp_tokens);

        response = response.substr(0, response.find(end_token));
        std::cout << "Predicted Response: \n" << response << std::endl;

        std::string actual_response = tok.detokenize(output_tokens[i]);

        actual_response = actual_response.substr(0, actual_response.find(end_token));
        
        std::cout << "Actual Response: \n" << actual_response << std::endl;

        std::cout << std::endl;
    }
    double total_loss = 0.0;
    for (const auto &loss_map : losses)
    {
        for (const auto &loss_entry : loss_map)
        {
            total_loss += loss_entry.second; // Add up all the individual losses
        }
    }
    double average_loss = total_loss / losses.size(); // Compute the average loss if needed
    std::cout << "Total Loss: " << total_loss << std::endl;
    std::cout << "Average Loss: " << average_loss << std::endl;

    return 0;
}
