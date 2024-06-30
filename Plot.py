import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def read_matrices_from_file(file_path):
    matrices = []
    current_matrix = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        # Remove any leading/trailing whitespace and split by semicolon
        row = line.strip().split(';')

        # Check for an empty row
        if all(elem.strip() == '' for elem in row):
            if current_matrix:
                matrices.append(current_matrix)
                current_matrix = []
            continue

        # Convert elements to floats (or you can use int if you prefer), skipping empty strings
        current_matrix.append([float(elem.strip()) for elem in row if elem.strip()])

    # Append the last matrix if there's no trailing empty row
    if current_matrix:
        matrices.append(current_matrix)

    return matrices

def process_and_plot_combined_matrices(file_paths):
    all_matrices = [read_matrices_from_file(file_path) for file_path in file_paths]
    max_matrices = max(len(matrices) for matrices in all_matrices)

    for i in range(max_matrices):
        combined_matrix = []

        for matrices in all_matrices:
            if i < len(matrices):
                combined_matrix.extend(matrices[i])

        if combined_matrix:
            print("Combined Matrix {}:".format(i+1))
            for row in combined_matrix:
                print(row)
            plot_matrix(combined_matrix)

def plot_matrix(matrix):
    matrix = np.array(matrix)

    # Create a custom colormap
    cmap = mcolors.ListedColormap(['purple', 'yellow'])
    bounds = [0, 0.5, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(matrix, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[0, 1, 2])

    plt.show()

file_paths = ['matrix_output_rank_0.txt', 'matrix_output_rank_1.txt', 'matrix_output_rank_2.txt']
process_and_plot_combined_matrices(file_paths)