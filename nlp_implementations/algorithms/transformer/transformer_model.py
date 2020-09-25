import torch
import torch.nn as nn
from .transformer_modules import TransformerDecoder, TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoderLayer, TokenDecoder, TokenEncoder


class TransformerModel(nn.Module):

    def __init__(self, nencoder_layers, ndecoder_layers, d_model, vocab, nheads, dim_feedforward = 2048, dropout= 0.1,
                 activation="relu"):
        super(TransformerModel, self).__init__()
        self.nencoder_layers = nencoder_layers
        self.ndecoder_layers = ndecoder_layers
        self.d_model = d_model
        self.vocab = vocab
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward;
        self.dropout = dropout;
        self.activation = activation;

        self.token_encoder = TokenEncoder(self.d_model, self.vocab, dropout = self.dropout)
        self.transformer_encoder_layer = TransformerEncoderLayer(self.d_model, self.nheads, dim_feedforward = self.dim_feedforward,
                                                                 dropout = self.dropout)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, self.nencoder_layers)
        self.transformer_decoder_layer = TransformerDecoderLayer(self.d_model, self.nheads, dim_feedforward = self.dim_feedforward,
                                                                 dropout = self.dropout, activation = self.activation)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, self.ndecoder_layers)
        self.token_decoder = TokenDecoder(self.d_model, self.vocab)

    def forward(self, src, target):
        src_encoded = self.token_encoder.forward(src)
        target_encoded = self.token_encoder.forward(target)

        memory = self.transformer_encoder.forward(src_encoded)
        output_encoded = self.transformer_decoder.forward(target_encoded, memory)

        output = self.token_decoder.forward(output_encoded)

        return output
