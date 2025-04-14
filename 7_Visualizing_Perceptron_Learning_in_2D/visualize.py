import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(output):
    return output * (1 - output)

# NAND gate dataset
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
epochs = 100
lines = []  # To store line per epoch

# Training loop and tracking decision boundary
for epoch in range(epochs):
    z = np.dot(X, weights) + bias
    output = sigmoid(z)
    error = y - output
    adjustments = error * sigmoid_derivative(output)
    weights += lr * np.dot(X.T, adjustments)
    bias += lr * np.sum(adjustments)

    # Store the current decision boundary line
    w0, w1 = weights.flatten()
    b = bias[0]
    x_vals = np.array([0, 1.2])
    y_vals = (-w0 * x_vals - b) / w1 if w1 != 0 else np.zeros_like(x_vals)
    lines.append((x_vals.copy(), y_vals.copy()))

# Set up the plot
fig, ax = plt.subplots()
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.2, 1.2)
ax.set_title("Perceptron Learning: NAND Gate")
ax.set_xlabel("Input 1")
ax.set_ylabel("Input 2")

# Plot input points
for i, label in enumerate(y):
    color = 'green' if label else 'red'
    ax.scatter(X[i, 0], X[i, 1], color=color, s=100)

line, = ax.plot([], [], 'b-', lw=2)

# Update function for animation
def update(frame):
    x_vals, y_vals = lines[frame]
    line.set_data(x_vals, y_vals)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(lines), interval=50, blit=True)

# Save as GIF
ani.save("nand_learning.gif", writer=PillowWriter(fps=20))
print("✅ Animation saved as 'nand_learning.gif'")
