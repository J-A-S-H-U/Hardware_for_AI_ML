import numpy as np

# Sigmoid activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# NAND gate data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[1], [1], [1], [0]])

# Initialize weights and bias
np.random.seed(1)
weights = np.random.rand(2, 1) - 0.5
bias = np.random.rand(1) - 0.5

# Training parameters
lr = 0.1
epochs = 100000
# Training loop
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X, weights) + bias
    output = sigmoid(z)

    # Compute error
    error = y - output

    # Backpropagation
    adjustments = error * sigmoid_derivative(output)
    weights += lr * np.dot(X.T, adjustments)
    bias += lr * np.sum(adjustments)

# Final output
print("Trained weights:", weights)
print("Trained bias:", bias)
print("Predictions after training:")
print(np.round(sigmoid(np.dot(X, weights) + bias), 2))
