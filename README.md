# Logistic Regression from Scratch

This project implements logistic regression from scratch using only NumPy, without relying on machine learning libraries like scikit-learn.

## Features

- Supports binary classification
- Uses sigmoid activation
- Implements cost function and gradient descent manually
- Accepts dataset input from a CSV file
- Includes data normalization
- Regularization can be added manually

## Usage

1. Prepare your dataset as a CSV file. The last column should be the binary target label (0 or 1).
2. Load the data using NumPy.
3. Run the script to train the logistic regression model.
4. Evaluate predictions based on a threshold of 0.5.

## Example

Suppose your CSV contains features like exam scores and a binary label indicating pass/fail. The model will learn to classify based on the input features.

## Requirements

- Python 3.x
- NumPy

## File Structure

- `logistic_regression.py`: Main implementation file
- `data.csv`: Input dataset (sample)
- `README.md`: This file

## Author

Bakir Bećić
