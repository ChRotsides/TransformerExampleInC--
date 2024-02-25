# Transformers from Scratch

This project is an educational exploration into the transformer model architecture, specifically focusing on the implementation of the feed-forward network component in C++. The objective is to shed light on the intricacies of transformer models, which are pivotal in advancing the field of natural language processing and beyond. By dissecting and reconstructing the feed-forward network from scratch, we delve into the core principles that underpin these powerful models, offering a hands-on approach to demystifying complex machine learning algorithms.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Structure](#structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Transformers have revolutionized the field of natural language processing and beyond. This project breaks down the complexities of transformers into digestible, implementable parts, offering a hands-on approach to learning. By reconstructing the architecture piece by piece, we gain insights into the operational dynamics and the pivotal role of each component.

## Features

- **Embedding Layer**: Maps tokens to high-dimensional vectors.
- **Self-Attention Mechanism**: Enables the model to weigh the significance of different words in a sentence.
- **Positional Encoding**: Injects information about the position of tokens in the sequence.
- **Encoder and Decoder**: Core components that process the input and generate the output, respectively.
- **Layer Normalization**: Stabilizes the learning process by normalizing the inputs across features.
- **Feed-Forward Network**: Applies further transformations to the data after attention computation.

## Installation

This project is built using C++. Ensure you have a C++ compiler installed on your system. Clone the repository to your local machine:

```bash
git clone https://github.com/ChRotsides/TransformerExampleInCpp.git
cd transformers-from-scratch
```

## Usage
To compile and run the project, navigate to the project directory and use the following commands:

```bash

g++ -o main main.cpp ./src/*.cpp -I ./include
./main
```
This will execute the main.cpp file, which orchestrates the entire transformer model process, from tokenizing inputs to generating predictions.

## Structure
The project is structured as follows:

include/: Contains all header files defining the classes and methods for various components like the embedding layer, self-attention mechanism, encoder, and decoder.
src/: Implementation of the class methods defined in the header files.
main.cpp: The entry point of the application, demonstrating the usage of different components.
## Contributing
Contributions to this project are welcome! Please fork the repository, make your changes, and submit a pull request.
