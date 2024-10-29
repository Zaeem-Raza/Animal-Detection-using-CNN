# Animal Detection using CNN with TensorFlow and Keras

This project demonstrates an implementation of an image classification model using Convolutional Neural Networks (CNN) with TensorFlow's Keras API. The CNN is designed to classify images into distinct categories by learning features through multiple layers of convolutions and pooling. This repository contains a Jupyter Notebook that trains the model on a dataset of 32x32 pixel images.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview

The notebook demonstrates how to build and compare two neural network architectures—an Artificial Neural Network (ANN) and a Convolutional Neural Network (CNN)—for image classification. The model was trained and tested on datasets similar to CIFAR-10, which contains 10 distinct classes (e.g., airplanes, cars, birds). The model utilizes convolutional and pooling layers to capture spatial features in the images, followed by dense layers to perform the final classification.

## Model Architecture

The architecture used in this project for the CNN model is as follows:

1. **Input Layer**: Accepts images of size 32x32 with 3 color channels (RGB).
2. **Convolutional Layers**:
    - **Conv2D**: 32 filters, 3x3 kernel, ReLU activation, followed by a 2x2 max pooling layer.
    - **Conv2D**: 64 filters, 3x3 kernel, ReLU activation, followed by a 2x2 max pooling layer.
3. **Flatten Layer**: Converts the 3D feature map from the convolutional layers into a 1D vector.
4. **Fully Connected Layers**:
    - **Dense**: 64 neurons, ReLU activation.
    - **Output Dense**: 10 neurons (for 10 classes), Softmax activation to output probabilities for each class.

This structure progressively reduces the spatial dimensions and learns hierarchical representations of the input images, which improves classification accuracy.

## Requirements

To run the notebook, the following libraries and tools are required:
- Python 3.x
- TensorFlow
- Matplotlib
- NumPy
- Jupyter Notebook or JupyterLab

These packages provide the necessary functions for building the CNN, visualizing the results, and running the notebook.

## Installation

1. **Clone the repository**: Download the repository files to your local machine.

2. **Install required packages**: Ensure all dependencies are installed. You can install them via pip.

## Usage

1. **Open the Notebook**:
   - Run `image_classification.ipynb` in Jupyter Notebook or JupyterLab.

2. **Load and Prepare Dataset**:
   - The notebook includes code to load and preprocess the CIFAR-10 dataset, but other datasets of 32x32 images can also be used by modifying the dataset loading cell.

3. **Run Each Cell**:
   - Follow the cells sequentially to define, compile, and train the model.
   - You can modify parameters like the number of epochs, batch size, and learning rate for experimentation.

4. **Evaluate the Model**:
   - The notebook includes code for evaluating the model’s performance on a test dataset and displays the model’s accuracy and loss over training epochs.
   - Sample predictions on test images are displayed to visualize the model's performance.

## Results

The project includes performance metrics for both the ANN and CNN models:

- **ANN Model**:
  - **Accuracy**: 55%

- **CNN Model**:
  - **Training Accuracy**: 79%
  - **Test Accuracy**: 69%

The CNN model achieves significantly higher accuracy compared to the ANN, as its convolutional layers are better suited for capturing spatial patterns in image data. The notebook also includes visualizations for tracking training and validation accuracy and loss, as well as code to display predictions on sample images.

