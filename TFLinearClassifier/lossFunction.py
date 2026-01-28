#Loss function 

def mean_squared_error(targets, prediction):
  per_sample_losses = tf.square(targets - prediction)

  return tf.reduce_mean(per_sample_losses)
