def naive_matrix_vector_product(x,y):
  #x is a NumPy matrix
  assert len(x.shape) ==2

  #y is a Numpy vector
  assert len(x.shape) == 1

  #The 1st dimension of x must equal the 0th dimension of y!

  assert x.shape[1] == y.shape[0]

  #This operation returns a vector of 0s with as many rows as x.
  z = np.zeros(x.shape[0])

  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      z[i] += x[i,j] * y[j]

      return z
