import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

def read_matrices_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip()
        matrices = content.split('\n\n')
        matrix_list = []
        for matrix in matrices:
            rows = matrix.split('\n')
            matrix_data = [list(map(float, row.split())) for row in rows]
            matrix_list.append(np.array(matrix_data))
        return matrix_list

def update_frame(num, matrices, img, title_prefix):
    img.set_array(matrices[num])
    img.axes.set_title(f'{title_prefix} t = {num}')
    return img,

def create_animation(matrices, title_prefix):
    # Custom colormap and normalization
    colors = ['green', 'lightcoral', 'coral', 'red', 'green', 'green']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots()
    img = ax.imshow(matrices[0], cmap=cmap, norm=norm, aspect='auto')
    ax.set_title(f'{title_prefix} t = 0')
    fig.colorbar(img, ax=ax, ticks=[0.0, 1.0, 2.0, 3.0])
    
    ani = animation.FuncAnimation(
        fig, update_frame, fargs=(matrices, img, title_prefix), frames=len(matrices), interval=500, blit=True
    )

    plt.show()

# File paths for the three text files
file_paths = ['infection_0.txt', 'infection_1.txt', 'infection_2.txt', 'infection_3.txt']

# Read and animate matrices from each file
for i, file_path in enumerate(file_paths):
    matrices = read_matrices_from_file(file_path)
    create_animation(matrices, f'File {i+1}')
