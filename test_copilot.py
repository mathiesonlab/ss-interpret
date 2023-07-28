
# write a function to compute the average of a list of numbers
def average(numbers):
    return sum(numbers) / len(numbers)

# write a function to open a file of floats with 4 columns, then compute the 
# average of the 3rd column
def average_third_column(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        numbers = []
        for line in lines:
            numbers.append(float(line.split()[2]))
        return average(numbers)