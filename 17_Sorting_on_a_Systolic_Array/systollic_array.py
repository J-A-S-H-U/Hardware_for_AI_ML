def systolic_array_sort(arr):
    N = len(arr)
    pe = arr.copy()  # Each PE holds one element

    for step in range(N):  # N steps are enough for bubble sort
        new_pe = pe.copy()  # To store updated values after step
        for i in range(N - 1):
            # Compare adjacent PEs and swap if needed
            if pe[i] > pe[i + 1]:
                new_pe[i], new_pe[i + 1] = pe[i + 1], pe[i]
        pe = new_pe  # Update the PE array
    return pe

# Example usage
data = [5, 1, 4, 2, 8]
sorted_data = systolic_array_sort(data)
print("Input:", data)
print("Sorted:", sorted_data)
