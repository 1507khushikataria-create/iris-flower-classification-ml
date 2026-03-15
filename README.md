# iris-flower-classification-ml
Machine learning project for classifying Iris flower species using Random Forest and Scikit-learn.
# Iris Flower Classification using K-Nearest Neighbors (KNN)

## Overview

This project builds a machine learning model to classify iris flowers into three species using the **K-Nearest Neighbors (KNN)** algorithm.
The model is trained using the well-known Iris dataset and demonstrates a complete machine learning workflow including data visualization, model training, and evaluation.

## Dataset

The dataset used in this project is the **Iris dataset**, available in scikit-learn.

The dataset contains **150 samples** of iris flowers with the following features:

* Sepal Length (cm)
* Sepal Width (cm)
* Petal Length (cm)
* Petal Width (cm)

Target Variable:

* Species of the iris flower

The dataset includes three classes:

* Setosa
* Versicolor
* Virginica

## Technologies Used

This project uses the following tools and libraries:

* Python
* NumPy
* pandas
* matplotlib
* seaborn
* scikit-learn

## Data Visualization

To understand the dataset, several visualizations were created:

* Scatter plot of **Sepal Length vs Sepal Width**
* Scatter plot of **Petal Length vs Petal Width**
* Pairplot visualization showing relationships between all features

These visualizations help observe how different species are distributed across the feature space.

## Machine Learning Model

The model used in this project is:

**K-Nearest Neighbors (KNN) Classifier**

Steps performed:

1. Load dataset
2. Convert dataset into a DataFrame
3. Visualize data
4. Define features and target variable
5. Split data into training and testing sets
6. Train the KNN model
7. Make predictions on test data

## Model Evaluation

The model performance is evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report**

These metrics help measure how well the model classifies the iris species.

## Project Structure

```
iris-flower-classification
│
├── iris_classification.py
└── README.md
```

## Result

The trained KNN model successfully classifies iris flowers into their respective species with high accuracy.

## Author

Machine Learning project created as part of a beginner ML portfolio.
