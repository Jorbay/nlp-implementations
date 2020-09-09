#Recommended stuff from Annotated Transformer
# Standard PyTorch imports
import torch
import torch.nn as nn
import math, copy
from torch.nn.modules.container import ModuleList
from torch import Tensor

#From torch documentation for api:
#https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
from typing import Optional

class Transformer():

    class TransformerEncoderLayer(nn.Module):
        """
        This is my  version of Pytorch's nn.TransformerEncoderLayer, which
        is made up of self-attn and feedforward network.
        In AIAYN, d_model is 512 and nhead is 8
        """

        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                     activation='relu'):
            super(Transformer.TransformerEncoderLayer, self).__init__()
            self.normalizer = nn.LayerNorm(d_model)
            self.multiHeadAttention = MultiHeadAttention(nhead, d_model)
            self.feedForwardNetwork = FeedForwardNetwork(d_model, dim_feedforward, d_model)

        def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                    src_key_padding_mask: Optional[torch.Tensor] = None):
            "Take in embedding and output hidden state + output of type torch.Tensor"
            result = self.multiHeadAttention.forward(src, src, src)
            result = self.__addAndNorm(result, src)

            feed_forward_result = self.feedForwardNetwork.forward(result)
            result = self.__addAndNorm(feed_forward_result, result)

            return result

        def __addAndNorm(self, ultima, penultima):
            sum = ultima + penultima
            return self.normalizer(sum)

    class TransformerEncoder(nn.Module):
        """
        This is my  version of Pytorch's nn.TransformerEncoder, which takes
        as argument an instance of an encoder_layer. The different layers of the
        encoder are then created with a copy operation. I'm going to use the same
        copy operation that pytorch uses for simplicity.
        """

        def __init__(self, encoder_layer, num_layers, norm=None):
            super(Transformer.TransformerEncoder, self).__init__()
            self.layers = self._get_clones(encoder_layer, num_layers)
            self.num_layers = num_layers

        def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
            output = src
            for layer in self.layers:
                output = layer.forward(output)
            return output

        # From Pytorch 1.6.0
        def _get_clones(self, module, N):
            return ModuleList([copy.deepcopy(module) for i in range(N)])


class SingleLayerPerceptron(nn.Module):
    def __init__(self, d_input, d_output):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(d_input, d_output)

    def forward(self, x):
        return self.fc(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dk = int(self.d_model / self.n_heads)
        self.v_linearizers = []
        self.q_linearizers = []
        self.k_linearizers = []
        self.o_linearizer = SingleLayerPerceptron(self.n_heads * self.dk, self.d_model)

        for i in range(0, self.n_heads):
            self.v_linearizers.append(SingleLayerPerceptron(self.d_model, self.dk))
            self.q_linearizers.append(SingleLayerPerceptron(self.d_model, self.dk))
            self.k_linearizers.append(SingleLayerPerceptron(self.d_model, self.dk))

    def forward(self, query, key, value):
        attentions = []

        for i in range(0, self.n_heads):
            current_v_linearizer = self.v_linearizers[i]
            current_q_linearizer = self.q_linearizers[i]
            current_k_linearizer = self.k_linearizers[i]

            linearized_query = current_q_linearizer(query)
            linearized_key = current_k_linearizer(key)
            linearized_value = current_v_linearizer(value)

            attentions.append(self.scaledDotProductAttention(
                linearized_query, linearized_key, linearized_value))

        concatenated_attentions = torch.cat(attentions, dim=1)
        result = self.o_linearizer(concatenated_attentions)
        return result

    def scaledDotProductAttention(query, key, value):
        result = torch.matmul(query, key.t())

        scale_factor = math.sqrt(int(query.size()[1]))
        result = result / scale_factor

        softmaxer = nn.Softmax(dim=0)
        result = softmaxer(result)

        result = torch.matmul(result, value)

        return result


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