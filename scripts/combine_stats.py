import numpy as np

def combine_npy_files(file1, file2, output_file):
    # Load the numpy arrays from the files
    array1 = np.load(file1)
    array2 = np.load(file2)
    print(array1.shape, array2.shape)
    
    # Check if the number of rows is the same
    if array1.shape[0] != array2.shape[0]:
        raise ValueError("The input files must have the same number of rows")
    
    # Concatenate the arrays along the columns
    combined_array = np.hstack((array1, array2))
    print(combined_array.shape)
    
    # Save the combined array to the output file
    np.save(output_file, combined_array)

# Example usage
def main():
    for pop in ["CEU", "CHB", "CHS", "ESN", "GBR", "YRI"]:
        combine_npy_files(f"stats_{pop}.npy", f"stats_{pop}_extra.npy", f"stats_{pop}_all.npy")

main()