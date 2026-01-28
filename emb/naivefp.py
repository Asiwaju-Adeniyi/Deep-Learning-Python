# keras.ops is where you will find all the tensor operations you need.

import keras 
from keras import ops 

class NaiveDense: 
  def __init_(self, input_size, output_size, activation=None):
    self.activation = activation
    self.W = keras.Variable(
        
      # Creates a matrix W of shape (input_size, output_size),
      # initialized with random values drawn from a uniform
      # distribution
        shape = (input_size, output_size), initializer = "uniform"
    )

        # Creates a vector b of shape (output_size,), initialized with
        # zeros

    self.b = keras.Variable(shape=(output_size,), intializer = "zeros")

        
        
    # Applies the forward pass
    def __call__(self, inputs):
      x = ops.matmul(inputs, self.W)
      x = x.self.b

      if self.activation is not None: 
        x = self.activation(x)

        return x


    # The convenience method for retrieving the layer's weights
        def weights(self):
          return [self.W, self.b]
