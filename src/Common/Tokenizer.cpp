// Tokenizer.cpp

#include "../../include/Common/Tokenizer.h"

Tokenizer::Tokenizer(){
    // Initialize the word_to_token_id map
    this->word_to_token_id = std::unordered_map<std::string, int>();
}

std::vector<int> Tokenizer::tokenize(std::string text){
    std::vector<int> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word)
    {
     
        // If the word is not in the map, add it
        if (this->word_to_token_id.find(word) == this->word_to_token_id.end())
        {
            int token_id = this->word_to_token_id.size();
            this->word_to_token_id[word] = token_id;
            this->token_id_to_word[token_id] = word;
        }
        tokens.push_back(this->word_to_token_id[word]);
    }
    return tokens;
}

void Tokenizer::printWordToTokenIdMap() const{
    std::cout << "Word to Token ID Map:" << std::endl;
    for (const auto &pair : this->word_to_token_id)
    {
        std::cout << pair.first << " -> " << pair.second << std::endl;
    }
}

int Tokenizer::getVocabSize(){
    return this->word_to_token_id.size();
}

std::string Tokenizer::detokenize(std::vector<int> tokens){
    std::string text;
    for (int i = 0; i < tokens.size(); i++)
    {
        text+=this->token_id_to_word[tokens[i]] + " ";
    }
    return text;
}

void Tokenizer::saveTokenMap(std::string filename){
    std::ofstream file;
    file.open(filename);
    for (const auto &pair : this->word_to_token_id)
    {
        file << pair.first << " " << pair.second << std::endl;
    }
    file.close();
}
void Tokenizer::loadTokenMap(std::string filename){
    std::ifstream file;
    try
    {
        
        file.open(filename);
    }
    catch(const std::exception& e)
    {
        return;
    }
    std::cout << "Loaded token map succesfully" << std::endl;
    std::string word;
    int token_id;
    while (file >> word >> token_id)
    {
        this->word_to_token_id[word] = token_id;
        this->token_id_to_word[token_id] = word;
    }
    file.close();
}
std::vector<std::vector<double>> Tokenizer::oneHotEncode(int token_id) {
    std::vector<std::vector<double>> oneHotLabels;
    std::vector<double> oneHotLabel(this->word_to_token_id.size(), 0);
    oneHotLabel[token_id] = 1;
    oneHotLabels.push_back(oneHotLabel);

    return oneHotLabels;
}

Tokenizer::~Tokenizer()
{
}