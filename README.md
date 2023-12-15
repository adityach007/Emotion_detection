# Deployment Link : https://emotion-detection-m7bn.onrender.com
# Emotion Classification from Audio using MLPClassifier

This repository contains code for classifying emotions from audio using Multi-Layer Perceptron (MLP) Classifier. The project utilizes the RAVDESS dataset, extracting features using Librosa and training a neural network for emotion classification.

## Introduction

The goal of this project is to classify emotions (such as calm, happy, fearful, disgust) from audio recordings using machine learning. We employ feature extraction techniques like MFCCs, chroma, and mel spectrograms to capture audio characteristics.

## Dataset and Preprocessing

The RAVDESS dataset comprises audio files labeled with emotions. We extract features, considering only selected emotions ('calm', 'happy', 'fearful', 'disgust'), using Librosa and process the data for training.

## Model Building and Training

The code builds MLPClassifier models with different hyperparameters to classify emotions. Two models are created, trained, and evaluated for accuracy.

1. **Model 1**: MLPClassifier with (alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), learning_rate='adaptive')
   - Achieved accuracy: 85.7%

2. **Model 2**: MLPClassifier with (alpha=0.001, batch_size=128, hidden_layer_sizes=(200, 200, 100, 50), learning_rate='adaptive', max_iter=500)
   - Achieved accuracy: 89.2%

## Evaluation

Both models are evaluated on the test set to measure their accuracy in emotion classification. Additionally, loss curves are plotted for visualizing model training progress.

## Saving the Model

The best-performing model (Model 2) is saved using Pickle for future use in emotion classification tasks.

## Usage

To use the trained model for emotion classification:
- Load the saved model ('emotion_classification-model.pkl').
- Provide audio files for emotion prediction using the loaded model.


# Detailed Explanation of the Code

## Feature Extraction Function (extract_feature)

This function extracts features from audio files. It takes a file name as input and extracts various features like MFCCs (Mel-frequency cepstral coefficients), chroma, and mel spectrograms using Librosa library functions.

## Loading Data Function (load_data)

This function loads the dataset by iterating through audio files in the provided directory. It filters specific emotions ('calm', 'happy', 'fearful', 'disgust') and extracts features using the extract_feature function. It splits the dataset into training and testing sets using train_test_split from scikit-learn.

## Model Creation and Training

Two MLPClassifier models are created with different hyperparameters. These models are trained using the training data obtained from the load_data function. Model 1 (model) and Model 2 (model1) are trained separately.

model: MLPClassifier with hyperparameters (alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), learning_rate='adaptive')

model1: MLPClassifier with hyperparameters (alpha=0.001, batch_size=128, hidden_layer_sizes=(200, 200, 100, 50), learning_rate='adaptive', max_iter=500)

## Model Evaluation

After training, both models are evaluated on the test set (x_test and y_test) to measure their accuracy using accuracy_score from scikit-learn. The accuracy of both models is printed out.

## Saving the Model

The best-performing model (model1) is saved using Pickle (pickle.dump) for future use in emotion classification tasks.

## Conclusion

The project demonstrates the use of audio processing and machine learning to classify emotions from audio recordings. It showcases how MLPClassifier models can be trained and used for emotion classification tasks.

---

The provided README gives an overview of the project, data preprocessing, model building, training, evaluation, and model saving steps. It aims to guide users on understanding the project's purpose, implementation, and usage of the trained model.

