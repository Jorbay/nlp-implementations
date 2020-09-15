import torch
import torch.nn as nn
import numpy as np

class SingleLayerPerceptron(nn.Module):
    def __init__(self, d_input, d_output):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(d_input, d_output)

    def forward(self, x):
        return self.fc(x)

# In AIAYN, 512 is used as input and output dimension, inner layer is 2048
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_input, d_hidden, d_output):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_input, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_output)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

def make_mask(number_of_tokens):
  #Following was from http://juditacs.github.io/2018/12/27/masked-attention.html
  #Annotated AIAYN recommends instead using np.triu
  #mask = torch.arange(number_of_tokens)[None, :] < mask_range[:, None]
  base_of_ones = np.ones((number_of_tokens, number_of_tokens))

  subsequent_mask = np.tril(base_of_ones, k=0).astype('uint8')
  return torch.from_numpy(subsequent_mask)
