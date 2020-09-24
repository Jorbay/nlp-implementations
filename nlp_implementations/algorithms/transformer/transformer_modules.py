#Recommended stuff from Annotated Transformer
# Standard PyTorch imports
import torch
import torch.nn as nn
import math, copy
from torch.nn.modules.container import ModuleList
from torch import Tensor
from .attention import MultiHeadAttention
from .toolbox import FeedForwardNetwork, make_positional_encoding
from .embedding import Embedding

#From torch documentation for api:
#https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
from typing import Optional

class TransformerEncoderLayer(nn.Module):
    """
    This is my  version of Pytorch's nn.TransformerEncoderLayer, which
    is made up of self-attn and feedforward network.
    In AIAYN, d_model is 512 and nhead is 8
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.normalizer = nn.LayerNorm(d_model)
        self.multiHeadAttention = MultiHeadAttention(nhead, d_model)
        self.feedForwardNetwork = FeedForwardNetwork(d_model, dim_feedforward, d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None):
        "Take in embedding and output hidden state + output of type torch.Tensor"
        result = self.multiHeadAttention.forward(src, src, src, src_mask)
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
        super(TransformerEncoder, self).__init__()
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer.forward(output, src_mask=mask)
        return output

    # From Pytorch 1.6.0
    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.normalizer = nn.LayerNorm(d_model)
        self.multiHeadAttention = MultiHeadAttention(nhead, d_model)
        self.feedForwardNetwork = FeedForwardNetwork(d_model, dim_feedforward, d_model)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # TGT is the target input
        # Memory is the hidden state from the encoder
        # TGT_mask is the mask needed for masked multi-head attention
        result = self.multiHeadAttention.forward(tgt, tgt, tgt, tgt_mask)
        result = self.__addAndNorm(result, tgt)

        intermediate_result = self.multiHeadAttention.forward(memory, memory, tgt)
        result = self.__addAndNorm(intermediate_result, result)

        feed_forward_result = self.feedForwardNetwork.forward(result)
        result = self.__addAndNorm(feed_forward_result, result)

        return result

    def __addAndNorm(self, ultima, penultima):
        sum = ultima + penultima
        return self.normalizer(sum)

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer.forward(tgt, memory, tgt_mask)
        return output

    # From Pytorch 1.6.0
    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for i in range(N)])

class TokenDecoder(nn.Module):
    def __init__(self, d_model, vocab):
        super(TokenDecoder, self).__init__()
        self.linear_layer = nn.Linear(d_model, vocab)
        self.softmax_layer = nn.Softmax(dim=0)

    def forward(self,x):
        linear_result = self.linear_layer(x)
        result = self.softmax_layer(linear_result)
        return result

class TokenEncoder(nn.Module):
    def __init__(self, d_model, vocab):
        super(TokenEncoder, self).__init__()
        self.embedder = Embedding(d_model, vocab)
        self.d_model = d_model
        self.vocab = vocab

    def forward(self, x):
        embedded = self.embedder.forward(x)
        number_of_tokens = embedded.size()[0]

        positional_encoding = make_positional_encoding(number_of_tokens, self.d_model)

        #Move positional encoding to same device as embedded tensor
        positional_encoding = embedded.new(positional_encoding)

        return embedded + positional_encoding