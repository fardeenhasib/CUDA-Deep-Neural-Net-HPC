# CUDA Neural Networks HPC project

This project demonstrates how to implement a binary number classifier using a CUDA neural network. The project consists of the following files:

- `CUDA Neural Networks HPC project.ipynb`: This notebook contains the instructions and explanations for the project.
- `nvcc4jupyter`: This is a Python package that allows running CUDA code in Jupyter notebooks. It can be installed using pip from GitHub.

The project requires a GPU with CUDA support and the following software:

- NVIDIA CUDA Toolkit 11.8
- Python 3.7 or higher
- Jupyter Notebook

To run the project, follow these steps:

1. Clone or download this repository to your local machine.
2. Open a terminal and navigate to the project folder.
3. Install `nvcc4jupyter` by running `pip install git+https://github.com/andreinechaev/nvcc4jupyter.git`.
4. Launch Jupyter Notebook by running `jupyter notebook`.
5. Open the `CUDA Neural Networks HPC project.ipynb` file and follow the instructions in the notebook.

The project will train a neural network to classify binary numbers from 0 to 15 into odd or even categories. The network will have one input layer, one hidden layer, and one output layer. The network will use sigmoid activation functions and backpropagation algorithm to learn from the data. The network will be tested on new binary numbers and its accuracy will be reported.

The project will also demonstrate how to use CUDA kernels to perform matrix operations and activation functions on the GPU, which can speed up the computation and improve the performance of the neural network.

The project is an example of how to use CUDA for high-performance computing (HPC) applications, such as machine learning and artificial intelligence.
