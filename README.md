# Fashion MNIST Hyperparameter Tuning Project

## Overview
This project focuses on building and optimizing a neural network model using **TensorFlow**, **Keras Tuner**, and the **Fashion MNIST dataset**. The primary goal is to tune hyperparameters such as the number of layers, units per layer, dropout rate, and learning rate to achieve optimal performance.

## Features
- **Dataset**: Uses the Fashion MNIST dataset for classification tasks.
- **Custom Model Architecture**:
  - Fully connected neural network with configurable hidden layers and units.
  - Normalization and dropout layers for regularization.
- **Hyperparameter Optimization**:
  - Implements Bayesian Optimization via Keras Tuner for parameter tuning.
  - Custom tuning logic for batch size selection.
- **Early Stopping**:
  - Prevents overfitting by monitoring validation accuracy.
- **Training**:
  - Conducts training with the best hyperparameters determined during tuning.

## Key Libraries
- **TensorFlow**: For building and training the neural network.
- **Keras Tuner**: For hyperparameter optimization.
- **Matplotlib**: For visualization (if needed).
- **NumPy**: For data manipulation.

## Project Structure
- **Data Loading**: Fashion MNIST dataset is loaded and preprocessed.
- **Model Creation**: 
  - A function `create_model` defines the architecture.
  - Configurable hyperparameters include:
    - Number of hidden layers
    - Number of units per layer
    - Dropout rate
    - Learning rate
- **Tuner Implementation**:
  - Custom tuner (`CustomTuner`) extends `kt.tuners.BayesianOptimization`.
  - Tuning searches for optimal parameters in a defined range.
- **Training**:
  - Trains the best model with a callback for early stopping.
- **Evaluation**: Validates the model on test data.

## Setup
### Prerequisites
Install the following Python packages:
- `tensorflow`
- `keras-tuner`
- `matplotlib`
- `numpy`

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install tensorflow keras-tuner matplotlib numpy
