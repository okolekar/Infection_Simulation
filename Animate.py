import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Step 1: Read the matrix from an Excel file
file_path = './time0.ods'  # Update this with the actual path to your Excel file
matrix_df = pd.read_excel(file_path, header=None)
matrix = matrix_df.to_numpy()

# Step 2: Define the color map
# Custom colors for the specified ranges
cmap = ListedColormap(['purple', 'yellow', 'purple'])

# Define the boundaries
bounds = [0, 1, 1.01, 3.1]

# Normalize the data to the bounds
norm = BoundaryNorm(bounds, cmap.N)

# Step 3: Plot the matrix
fig, ax = plt.subplots()
cbar = ax.imshow(matrix, cmap=cmap, norm=norm)

# Adding a color bar for reference
cbar = plt.colorbar(cbar, ticks=[0, 1, 1.01, 3])
cbar.ax.set_yticklabels(['0', '1', '1.01', '3'])

plt.show()