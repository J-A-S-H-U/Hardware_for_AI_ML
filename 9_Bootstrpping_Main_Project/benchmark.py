import numpy as np
import pandas as pd
import psutil
import time
import os
from matplotlib import pyplot as plt
from memory_profiler import profile
import tracemalloc


data = pd.read_csv(r'C:\Users\jaswa\Hardware_for_AI_ML\9_Bootstrpping_Main_Project\train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# === Benchmarking ===
tracemalloc.start()
@profile
def benchmark_training(X, Y, X_dev, Y_dev, alpha=0.10, iterations=500):
    print("\n=== Benchmarking Started ===")
    start_time = time.time()
    process = psutil.Process(os.getpid())
    cpu_percent_before = psutil.cpu_percent(interval=None)
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB

    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 100 == 0 or i == iterations - 1:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            print(f"Iteration {i}: Accuracy = {acc:.4f}")

    end_time = time.time()
    duration = end_time - start_time

    cpu_percent_after = psutil.cpu_percent(interval=None)
    mem_after = process.memory_info().rss / (1024 ** 2)  # MB

    # Evaluate on dev set
    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    accuracy = get_accuracy(dev_predictions, Y_dev)
    throughput = X_dev.shape[1] / duration

    print("\n=== Benchmarking Results ===")
    print(f"Latency (Training Time): {duration:.2f} seconds")
    print(f"CPU Usage: {cpu_percent_after}%")
    print(f"Memory Usage: {mem_after - mem_before:.2f} MB")
    print(f"Output Accuracy (Dev Set): {accuracy * 100:.2f}%")
    print(f"Throughput: {throughput:.2f} samples/sec")
    return W1, b1, W2, b2



# === Run Training with Benchmark ===
W1, b1, W2, b2 = benchmark_training(X_train, Y_train, X_dev, Y_dev)

snapshot = tracemalloc.take_snapshot()
tracemalloc.stop()
top_stats = snapshot.statistics('lineno')

with open('memory_usage.txt', 'w') as f:
    f.write("[ Top 10 memory consumers ]\n")
    for stat in top_stats[:10]:
        f.write(f"{stat}\n")

    f.write("\n[ Detailed traceback for the top memory consumer ]\n")
    for stat in top_stats[:1]:
        f.write('\n'.join(stat.traceback.format()) + '\n')

print("Memory usage details saved to 'memory_usage.txt'")