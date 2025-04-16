import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# XOR data
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [1], [1], [0]])

# Fix random seed for reproducibility
np.random.seed(1)

# Architecture sizes
input_size = 2
hidden_size = 2
output_size = 1

# Xavier initialization
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
b2 = np.zeros((1, output_size))

# Training parameters
lr = 0.5
epochs = 50000

# Training loop
for epoch in range(epochs):
    # ---- Forward Pass ----
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)

    # ---- Backpropagation ----
    error = y - output
    d_output = error * sigmoid_derivative(output)

    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(a1)

    # ---- Update weights and biases ----
    W2 += a1.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr

    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # Print loss occasionally
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} - Loss: {loss:.6f}")

# ---- Final Output ----
print("\nFinal predictions:")
for i, (inp, pred) in enumerate(zip(X, output)):
    print(f"Input: {inp} → Predicted: {pred[0]:.4f}")
