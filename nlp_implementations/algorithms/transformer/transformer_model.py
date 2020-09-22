import torch
import torch.nn as nn
from .transformer_modules import TransformerDecoder, TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoderLayer, TokenDecoder, TokenEncoder


class TransformerModel(nn.Module):
    pass