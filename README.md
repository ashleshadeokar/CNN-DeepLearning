# CNN-DeepLearning

This project aims to identify various plant diseases using Convolutional Neural Networks (CNNs). The dataset used for training and testing the model is the PlantVillage Dataset sourced from Kaggle. The project includes a deep learning model and a Streamlit-based web application for user-friendly disease detection.

## Table of Contents
- Project Overview
- Dataset
- Model Training
- Web Application
- How to Run

## Project Overview
Plant diseases can cause significant crop loss and economic damage. This project leverages the power of deep learning to classify plant diseases from images using a CNN-based model. It aims to assist farmers and researchers in identifying plant diseases early to enable timely intervention.

## Dataset
- The dataset used in this project is the PlantVillage Dataset, which can be found on Kaggle: PlantVillage Dataset.
- It consists of multiple categories of plant diseases, including healthy plants.
- The dataset is pre-processed and split into training, validation, and test sets for optimal model performance.

## Model Training
The model training was done using a Convolutional Neural Network (CNN) implemented in Python. The Jupyter notebook Plant_Disease_Prediction_CNN_Image_Classifier.ipynb contains the full process:

## Data Preprocessing: 
- Loaded the dataset, applied data augmentation, and normalized the images.
- Model Architecture: Utilized a CNN model with multiple convolutional layers, followed by max-pooling, dropout, and fully connected layers.
- Training: The model was trained on the Kaggle dataset using TensorFlow/Keras.
- Evaluation: Evaluated the model using test data, achieving high accuracy in detecting plant diseases.

## Web Application
A Streamlit application was built to provide an easy-to-use interface for users to upload images of plant leaves and get predictions on whether they are diseased or healthy.

### Features:
- Upload an image of a plant leaf.
- The model classifies the disease and displays the result.
- Provides confidence scores for the predictions.
The Streamlit app code is located in the main.py file. Configuration files such as config.toml and credentials.toml are used to store app settings.

## How to Run
Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Streamlit
- Docker (optional, for containerized deployment)

- Kaggle Dataset:  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
