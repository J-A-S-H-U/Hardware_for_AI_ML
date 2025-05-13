import numpy as np
import time
import matplotlib.pyplot as plt

def systolic_bubble_sort(arr):
    N = len(arr)
    pe = arr.copy()

    for step in range(N):
        for i in range(step % 2, N - 1, 2):  # even and odd phases
            if pe[i] > pe[i + 1]:
                pe[i], pe[i + 1] = pe[i + 1], pe[i]
    return pe

# Test it
print("Sorted:", systolic_bubble_sort([5, 2, 9, 1, 5, 6]))

def benchmark():
    sizes = [10, 100, 1000, 5000]
    times = []

    for size in sizes:
        arr = np.random.randint(0, 10000, size)
        start = time.time()
        systolic_bubble_sort(arr)
        end = time.time()
        times.append(end - start)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o')
    plt.xlabel("Input Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Systolic Bubble Sort Performance")
    plt.grid(True)
    plt.show()

benchmark()
