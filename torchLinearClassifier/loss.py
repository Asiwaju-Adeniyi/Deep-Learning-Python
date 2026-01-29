def mean_squared_error(targets, predictions): 
  per_sample_losses = torch.square(targets - predictions)
     return torch.mean(per_sample_losses) 
