def compute_pi(matrix):
    from random import random

    inside = 0
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if (i - 0.5) ** 2 + (j - 0.5) ** 2 <= 0.25:
                inside += 1
            total += 1
    return 4 * inside / total

# write a function to compute the average number of differences 
# between all pairs of rows in a 2D matrix
def average_differences(matrix):
    total = 0
    count = 0
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            for k in range(len(matrix[i])):
                if matrix[i][k] != matrix[j][k]:
                    total += 1
            count += 1
    return total / count
