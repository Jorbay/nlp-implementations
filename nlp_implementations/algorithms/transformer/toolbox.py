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

def make_positional_encoding(number_of_tokens, d_model):
    #The following is defined in section 3.5 from AIAYN
    position_tensor = torch.arange(end=number_of_tokens)
    position_tensor = torch.transpose(position_tensor.unsqueeze(0), 0, 1)
    position_tensor = position_tensor.repeat(1, d_model)

    dimension_tensor = torch.arange(end=d_model)
    dimension_tensor = dimension_tensor.repeat(number_of_tokens, 1)

    angles = torch.div(position_tensor, torch.pow(10000, torch.true_divide(
        2 * dimension_tensor, d_model)))

    encodings = angles.clone()
    # Apply sin to even dimensions and cos to odd dimensions
    encodings[:, 0::2] = torch.sin(encodings[:, 0::2])
    encodings[:, 1::2] = torch.cos(encodings[:, 1::2])

    return encodings