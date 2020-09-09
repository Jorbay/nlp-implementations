import torch.nn as nn

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