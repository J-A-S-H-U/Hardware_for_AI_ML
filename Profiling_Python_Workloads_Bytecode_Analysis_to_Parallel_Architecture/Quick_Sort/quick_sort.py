import random

def quicksort(arr):
    """
    Sorts the list 'arr' using the Quicksort algorithm and returns the sorted list.
    
    Args:
        arr (list): List of elements to be sorted.
    
    Returns:
        list: A new sorted list.
    """
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)

if __name__ == "__main__":
    # Generate a list of 50 random integers between 0 and 100
    random_list = [random.randint(0, 100) for _ in range(50)]
    print("Unsorted list:")
    print(random_list)
    
    sorted_list = quicksort(random_list)
    print("\nSorted list:")
    print(sorted_list)