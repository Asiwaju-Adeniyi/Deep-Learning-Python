#forward pass

def model (inputs, W, b): 
  return tf.matmul(inputs, W) + b
