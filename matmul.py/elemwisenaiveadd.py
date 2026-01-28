def naive_add(x,y): 
  #x and y are rank-2 Numpy tensors.

  assert len(x.shape) == 2
  assert x.shape == y.shape
  #Avoids overwriting the input tensor

  x = x.copy() 

  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i,j] += y[i,j]

      return x
