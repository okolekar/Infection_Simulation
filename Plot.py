import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np

def read_matrices_from_file(file_path):
    matrices = []
    current_matrix = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        row = line.strip().split(';')

        if all(elem.strip() == '' for elem in row):
            if current_matrix:
                matrices.append(current_matrix)
                current_matrix = []
            continue

        current_matrix.append([float(elem.strip()) for elem in row if elem.strip()])

    if current_matrix:
        matrices.append(current_matrix)

    return matrices

def process_and_plot_combined_matrices(file_paths):
    all_matrices = [read_matrices_from_file(file_path) for file_path in file_paths]
    max_matrices = max(len(matrices) for matrices in all_matrices)

    fig, ax = plt.subplots()

    ims = []
    for i in range(max_matrices):
        combined_matrix = []

        for matrices in all_matrices:
            if i < len(matrices):
                combined_matrix.extend(matrices[i])

        if combined_matrix:
            im = plot_matrix(ax, combined_matrix)
            ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True)
    plt.show()

def plot_matrix(ax, matrix):
    matrix = np.array(matrix)

    cmap = mcolors.ListedColormap(['purple', 'yellow'])
    bounds = [0, 0.5, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm)
    return im

file_paths = ['matrix_output_rank_0.txt', 'matrix_output_rank_1.txt', 'matrix_output_rank_2.txt']
process_and_plot_combined_matrices(file_paths)
