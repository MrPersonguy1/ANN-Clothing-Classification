👕 ANN Clothing Image Classification
This project uses an Artificial Neural Network (ANN) to recognize different types of clothing from the Fashion MNIST dataset. The goal is to teach the model to identify items like shirts, shoes, and jackets from grayscale images.

📚 What You’ll Learn
How to work with the Fashion MNIST dataset

How to build an ANN using PyTorch

How to train and evaluate a neural network on image data

How to visualize predictions and model performance

🛠️ Technologies Used
Python 🐍

PyTorch (for building and training the ANN)

torchvision (for dataset handling)

Matplotlib (for graphs and visualizations)

NumPy (for math and array handling)

🧠 How It Works
The Fashion MNIST dataset is loaded, containing 28x28 grayscale images of clothes.

A dictionary maps each class number to a clothing type (like "Sneaker" or "T-shirt").

The images and labels are loaded into data loaders.

An ANN model is created with hidden layers and ReLU activation functions.

The model is trained for multiple epochs to minimize error using cross-entropy loss.

Accuracy is calculated on the test set, and results are visualized.

🚀 How to Run
Make sure Python is installed.

Install the required packages:

bash
Copy
Edit
pip install torch torchvision matplotlib numpy
Run the script in a Jupyter Notebook or Python file.

The model will train and show accuracy graphs and example predictions.

📁 File Overview
ANN_Clothing_Classification.py: Full script to load data, define and train the ANN, and test its performance on clothing image data.

