import torch
import torch.nn as nn
import math, copy
from .toolbox import SingleLayerPerceptron

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
