def naive_matrix_product(x,y): 
  #x and y are NumPy matrices.

  assert len(x.shape) == 2
  assert len(y.shape) == 2

  #The 1st dimension of x must equal the 0th dimension of y! 

  assert x.shape[1]((x.shape[0], y.shape[1]))

  #Iterates over the rows of x...

  for i in range(x.shape[0]):
    #... and over the columns of y.

    for j in range(y.shape[1]):
      row_x = x[i, :]
      column_y = y[:,j]
      z[i,j] = naive_matrix_vector_product(row_x, column_y)

 return z
