import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (not used here, but useful for training)
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and bias
weights = np.array([0.4, -0.7])
bias = 0.1

# Number of iterations
iterations = 10  # You can change this to any number

# Loop for random inputs
for i in range(iterations):
    # Generate random inputs between 0 and 1
    inputs = np.random.rand(2)

    # Compute weighted sum
    z = np.dot(inputs, weights) + bias

    # Activation output
    output = sigmoid(z)

    # Print results
    print(f"Iteration {i+1}: Inputs = {inputs}, Output = {output:.4f}")