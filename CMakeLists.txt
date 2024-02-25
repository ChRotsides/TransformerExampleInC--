cmake_minimum_required(VERSION 3.10)  # Minimum version of CMake required

# Set the project name and version
project(TransformerExample VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
add_executable(main
    main.cpp
    FinalLayer.cpp
    Embedding.cpp
    Tokenizer.cpp
    ./Encoding/PositionalEncoding.cpp
    ./Encoding/Encoder.cpp
    ./Encoding/EncoderLayer.cpp
    Decoder.cpp
    DecoderLayer.cpp
    SelfAttention.cpp
    FeedForwardNetwork.cpp
    LayerNorm.cpp
)
