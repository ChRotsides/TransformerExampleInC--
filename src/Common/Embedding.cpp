// Embedding.cpp
#include "../../include/Common/Embedding.h"

Embedding::Embedding(int vocab_size, int embed_dim){
    this->vocab_size = vocab_size;
    this->embed_dim = embed_dim;
    
    
    this->embedding_matrix = generateRandomEmbeddingMatrix(embed_dim, vocab_size);
    // print the vector dims
    std::cout << "Embedding matrix dims: " << embedding_matrix->size() << " x " << embedding_matrix[0].size() << std::endl;
}

void Embedding::printEmbeddingMatrix() {
    std::cout << "Embedding matrix: " << std::endl;
    for (int i = 0; i < this->vocab_size; i++) { // Ensure loops are over vocab_size and embed_dim in the correct order
        for (int j = 0; j < this->embed_dim; j++) {
            // Set precision to 2 decimal places and a fixed width for alignment
            std::cout << std::fixed << std::setprecision(2) << std::setw(6) << (*this->embedding_matrix)[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void Embedding::saveEmbeddingMatrix(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    std::ostringstream oss;
    for (int i = 0; i < this->vocab_size; i++) {
        for (int j = 0; j < this->embed_dim; j++) {
            oss << (*this->embedding_matrix)[i][j] << " ";
        }
        oss << '\n'; // Use '\n' instead of std::endl to avoid flushing the stream unnecessarily
    }

    file << oss.str(); // Write the entire matrix to the file in one operation
    file.close();
}
void Embedding::loadEmbeddingMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    // Resize the embedding matrix to match the expected dimensions
    (*this->embedding_matrix).resize(this->vocab_size, std::vector<double>(this->embed_dim, 0.0f));

    std::string line;
    int row = 0;
    while (std::getline(file, line) && row < this->vocab_size) {
        std::istringstream iss(line);
        double value;
        int col = 0;

        while (iss >> value && col < this->embed_dim) {
            (*this->embedding_matrix)[row][col] = value;
            ++col;
        }

        if (col != this->embed_dim) {
            std::cerr << "Error: Line " << (row + 1) << " does not have enough values." << std::endl;
            return;
        }

        ++row;
    }

    if (row != this->vocab_size) {
        std::cerr << "Error: File does not have enough lines." << std::endl;
        return;
    }

    file.close();
}

std::vector<std::vector<double>> *Embedding::generateRandomEmbeddingMatrix(int embed_dim, int vocab_size){
    std::vector<std::vector<double>>* embedding_matrix= new std::vector<std::vector<double>>(vocab_size, std::vector<double>(embed_dim, 0.0));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < this->vocab_size; i++){
        for (int j = 0; j < this->embed_dim; j++){
            (*embedding_matrix)[i][j] = dis(gen);
        }
    }
    return embedding_matrix;
}

std::vector<double> Embedding::getEmbedding(int token_id){
    return (*this->embedding_matrix)[token_id];
}

std::vector<std::vector<double>> *Embedding::tokenToEmbeddings(std::vector<int> tokens){
    std::vector<std::vector<double>>* embeddings = new std::vector<std::vector<double>>( tokens.size() , std::vector<double>(this->embed_dim, 0.0));
    for (int i = 0; i < tokens.size(); i++){
        (*embeddings)[i] = (*this->embedding_matrix)[tokens[i]];
    }
    return embeddings;
}

Embedding::~Embedding()
{
    delete this->embedding_matrix;

}
