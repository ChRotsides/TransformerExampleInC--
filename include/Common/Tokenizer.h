// Tokenizer.h
#ifndef SELF_TOKENIZER_H
#define SELF_TOKENIZER_H

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <fstream>




class Tokenizer
{
private:
    std::unordered_map<std::string, int> word_to_token_id; // Maps words to unique token IDs
    std::unordered_map<int, std::string> token_id_to_word; // Maps token IDs to words
    int vocab_size;
public:
    Tokenizer();
    std::vector<int> tokenize(std::string text);
    int getVocabSize();
    void printWordToTokenIdMap() const;
    std::string detokenize(std::vector<int> tokens);
    void saveTokenMap(std::string filename);
    void loadTokenMap(std::string filename);
    std::vector<std::vector<double>> oneHotEncode(int token_id);
    ~Tokenizer();
};

#endif