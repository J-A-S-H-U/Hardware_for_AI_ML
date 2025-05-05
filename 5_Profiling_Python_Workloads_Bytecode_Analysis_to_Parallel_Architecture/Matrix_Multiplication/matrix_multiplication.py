import random

def generate_matrix(rows, cols):
    """Generates a matrix with random integers between 0 and 9."""
    return [[random.randint(0, 9) for _ in range(cols)] for _ in range(rows)]

def multiply_matrices(A, B):
    """Multiplies two matrices A and B."""
    n = len(A)          # Number of rows in A
    m = len(B[0])       # Number of columns in B
    p = len(A[0])       # Number of columns in A / rows in B

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(m)] for _ in range(n)]
    
    # Perform multiplication
    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Generate two 10x10 matrices
A = generate_matrix(10, 10)
B = generate_matrix(10, 10)

# Multiply the matrices
result = multiply_matrices(A, B)

# Display the matrices
print("Matrix A:")
for row in A:
    print(row)

print("\nMatrix B:")
for row in B:
    print(row)

print("\nResultant Matrix (A x B):")
for row in result:
    print(row)