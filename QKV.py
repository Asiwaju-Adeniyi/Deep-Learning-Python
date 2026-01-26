import numpy as np

#simple word embeddings (4-dimensions)

cat = np.array([1.0, 0.0, 0.5, 0.2])
eats = np.array([0.0, 1.0, 0.3, 0.8])
fish = np.array([0.5, 0.3, 1.0, 0.1])

#Stack all as input matrix
#Shape: (sequence_length, embedding_dim) = (3,4)

X = np.array([cat, eats, fish])
print("Input matrix X:")
print(X)
print(f"Shape: {X.shape}")
