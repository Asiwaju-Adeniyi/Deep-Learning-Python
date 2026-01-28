def naive_relu(x):

  #x is a rank-2 NumPy tensor.

  assert len(x.shape)==2

  #Avoids overwriting the input tensor
  x = x.copy()

  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      x[i,j] = max(x[i,j],0)

  return x
