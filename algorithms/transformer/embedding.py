import torch.nn as nn
import math

#The following is taken from the Annotated Transformer
#This class can also be modeled as a single layer perceptron. Ultimately, since
#inputs are converted into 1 hot encodings, the multiplication between the
#underlying weights matrix and the inputs is really just a lookup of a row of
#the  weight matrix.
class Embedding(nn.Module):
  def __init__(self, d_model, vocab):
    super(Embedding, self).__init__()
    self.lookup_table = nn.Embedding(vocab, d_model)
    self.d_model = d_model

  def forward(self, x):
    #the math.sqrt(self.d_model) was recommended by "Attention is all you need"
    #TODO: find out why this math.sqrt(self.d_model) is used.
    return self.lookup_table(x) * math.sqrt(self.d_model)