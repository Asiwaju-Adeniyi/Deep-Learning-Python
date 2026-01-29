class LinearModel(torch.nn.Module): 
  def __init__ (self):
    super().__init__()
    self.W = torch.nn.Parameter(torch.rand(input_dim, output_dim))
    self.b = torch.nn.Parameter(torch.rand(output_dim))

    def forward(self, inputs):
      return torch.matmul(inputs, self.W) + self.b
