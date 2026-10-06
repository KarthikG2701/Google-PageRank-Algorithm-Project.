import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix

# Task 1: Load the network data and check sparsity
data = scipy.io.loadmat("AdjMatrix.mat")
AdjMatrix = csr_matrix(data['AdjMatrix'])

num_elements = AdjMatrix.shape[0] * AdjMatrix.shape[1]
num_non_zero_elements = AdjMatrix.nnz
nnzAdjMatrix = num_non_zero_elements / num_elements
print(f"Sparsity of AdjMatrix: {nnzAdjMatrix:.4f}")

# Task 2: Check the dimensions of the matrix
m, n = AdjMatrix.shape
print(f"Dimensions of AdjMatrix: {m} x {n}")

# Task 3: Create a smaller submatrix and plot the network
NumNetwork = 500
AdjMatrixSmall = AdjMatrix[:NumNetwork, :NumNetwork].toarray()

# Generate random coordinates for the nodes
coordinates = np.random.rand(NumNetwork, 2) * NumNetwork

# Plot the graph
plt.figure(figsize=(10, 10))
plt.plot(coordinates[:, 0], coordinates[:, 1], 'k-*')
plt.title('Subgraph of the First 500 Nodes')
plt.xlabel('Random X Coordinate')
plt.ylabel('Random Y Coordinate')
plt.show()