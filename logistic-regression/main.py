import numpy as np
import pandas as pd

# Load data from CSV file
data = pd.read_csv('students.csv')

# Split features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
m, n = X.shape

# Feature scaling (standardization)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Initialize parameters
w = np.zeros(n)
b = 0

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (log loss)
def compute_cost(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)
    cost = -(1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return cost

# Compute gradients for w and b
def compute_gradients(X, y, w, b):
    m = X.shape[0]
    z = np.dot(X, w) + b
    h = sigmoid(z)
    error = h - y
    dw = (1/m) * np.dot(X.T, error)
    db = (1/m) * np.sum(error)
    return dw, db

# Training loop using gradient descent
def train(X, y, w, b, alpha, epochs):
    for i in range(epochs):
        dw, db = compute_gradients(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        if i % 100 == 0:
            cost = compute_cost(X, y, w, b)
            print(f"Epoch {i}: Cost = {cost:.4f}")
    return w, b

# Prediction function (returns True/False)
def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z) >= 0.5

# Train the model
alpha = 0.1
epochs = 1000
w, b = train(X, y, w, b, alpha, epochs)

# Evaluate model accuracy
y_pred = predict(X, w, b)
accuracy = np.mean(y_pred == y)
print(f"Model accuracy: {accuracy * 100:.2f}%")
