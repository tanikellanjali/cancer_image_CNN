# Skin Lesion Classification Using CNN
## Project Overview

This project focuses on building a Convolutional Neural Network (CNN) to classify skin lesion images into two categories: malignant and benign. The goal is to assist in the early detection of skin cancer by leveraging deep learning techniques to analyze medical images.

## Dataset

The dataset is structured as follows:
`train_cancer/benign/ image1.jpg,`
`train_cancer/benign/ image2.jpg,`
`train_cancer/malignant/image1.jpg,`
`train_cancer/malignant/image2.jpg,`
`train_cancer/.DS_Store (to be ignored)`

**benign/** - Contains images of benign skin lesions.

**malignant/** - Contains images of malignant skin lesions.
## Model Architecture

The CNN model consists of:

- Conv2D layers – Extracts features from the images.

- MaxPooling2D layers – Reduces dimensionality and computational complexity.

- Flatten layer – Converts 2D matrixes to 1D for dense layers.

- Dense layers – Fully connected layers for classification.

- Softmax activation – Classifies images into two categories.

## Preprocessing and Data Augmentation

- Rescaling: Pixel values are rescaled by 1/255.

- Augmentation: Rotation, zoom, width/height shift, and horizontal flip are applied to improve generalization.

- Validation Split: 20% of the data is reserved for validation.


## Training

The model is trained over 10 epochs using the categorical cross-entropy loss and Adam optimizer. Training and validation accuracy are monitored to ensure the model is learning effectively.

## Visualization

Sample images from each category are plotted to visualize the dataset.

Predictions are visualized with confidence scores to analyze model performance.

## How to Run the Project

- Clone the repository.

- Place the dataset in the ~/Downloads/train_cancer/ directory.

- Ensure dependencies are installed:

`pip install tensorflow matplotlib numpy pillow`

- Run the Jupyter notebook or Python script to train and evaluate the model.
## Potential Errors and Fixes

Class Mismatch Error: Exclude unwanted directories like .DS_Store and .ipynb_checkpoints.

Shape Mismatch Error: Ensure the dataset folders contain exactly two classes (benign and malignant).

## Results

The model achieves over 90% accuracy on validation data. Visualizations display predictions with confidence levels.

## Future Improvements

- Increase dataset size.

- Experiment with more complex architectures (ResNet, EfficientNet).

- Apply transfer learning for better performance.

## Acknowledgments

- Dataset curated for medical image classification research.

- TensorFlow and Keras for deep learning framework
