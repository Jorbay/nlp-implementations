import torch
import torch.nn as nn
from .transformer_modules import TransformerDecoder, TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoderLayer, TokenDecoder, TokenEncoder


class TransformerModel(nn.Module):

    def __init__(self, ntokens_per_batch, nencoder_layers, ndecoder_layers, d_model, nheads, vocab):
        super(TransformerModel, self).__init__()
        self.ntokens_per_batch = ntokens_per_batch
        self.nencoder_layers = nencoder_layers
        self.ndecoder_layers = ndecoder_layers
        self.d_model = d_model
        self.nheads = nheads
        self.vocab = vocab

        self.token_encoder = TokenEncoder(self.d_model, self.vocab)
        self.transformer_encoder_layer = TransformerEncoderLayer(self.d_model, self.nheads)
        self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, self.nencoder_layers)
        self.transformer_decoder_layer = TransformerDecoderLayer(self.d_model, self.nheads)
        self.transformer_decoder = TransformerDecoder(self.transformer_decoder_layer, self.ndecoder_layers)
        self.token_decoder = TokenDecoder(self.d_model, self.vocab)

    def forward(self, src):
        src_encoded = self.token_encoder.forward(src)

        #TODO: I need to also make an encoding of expected output (src shifted)
        target = None

        memory = self.transformer_encoder.forward(src)
        output_encoded = self.transformer_decoder.forward(target, memory)

        output = self.token_decoder.forward(output_encoded)
