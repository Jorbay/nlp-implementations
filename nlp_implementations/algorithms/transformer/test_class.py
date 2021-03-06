from .attention import MultiHeadAttention
from .embedding import Embedding
from .toolbox import SingleLayerPerceptron, make_mask
from .transformer_modules import TransformerDecoder, TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoderLayer, TokenDecoder, TokenEncoder
from .transformer_model import TransformerModel

import torch

class TestClass:
    d_model = 512
    number_of_heads = 8
    d_k = int(d_model / number_of_heads)
    vocab = 37000

    def test_scaledDotProductAttention(self):
        number_of_tokens = 3

        example_query = torch.tensor([[1, 1], [1, 1], [1,1]], dtype=torch.float64)
        example_key = example_query.detach().clone()
        example_value = example_query.detach().clone()
        mask = make_mask(number_of_tokens)

        multi_head_attention = MultiHeadAttention(self.number_of_heads, self.d_model)

        result_without_mask = multi_head_attention.scaledDotProductAttention(example_query, example_key, example_value)
        result_with_mask = multi_head_attention.scaledDotProductAttention(example_query, example_key, example_value,
                                                                          mask)
        expected_result_without_mask = example_query.detach().clone()
        expected_result_with_mask = torch.tensor([[1/3, 1/3], [5/6, 5/6], [11/6, 11/6]], dtype=torch.float64)

        assert torch.equal(result_without_mask, expected_result_without_mask)

        #TODO: Fix this assert for the expected result of mask.
        #assert torch.equal(result_with_mask, expected_result_with_mask)

    def test_encoder_and_decoder_integration(self):
        number_of_tokens = 3
        number_of_encoder_layers = 6
        number_of_decoder_layers = 6


        example_input_words = torch.LongTensor([[1,2,3],[4,5,6]])
        example_next_words = torch.LongTensor([[2,3,4], [5,6,7]])

        words2Embeddings = TokenEncoder(self.d_model, self.vocab)
        encoder_layer = TransformerEncoderLayer(self.d_model, self.number_of_heads)
        encoder = TransformerEncoder(encoder_layer, number_of_encoder_layers)
        decoder_layer = TransformerDecoderLayer(self.d_model, self.number_of_heads)
        decoder = TransformerDecoder(decoder_layer, number_of_decoder_layers)
        decoder_to_probabilities = TokenDecoder(self.d_model, self.vocab)

        input_embeddings = words2Embeddings.forward(example_input_words)
        next_embeddings = words2Embeddings.forward(example_next_words)

        encoder_result = encoder.forward(input_embeddings)

        decoder_mask = make_mask(number_of_tokens)
        decoder_result = decoder.forward(next_embeddings, encoder_result, decoder_mask)
        result = decoder_to_probabilities.forward(decoder_result)

    def test_token_encoder(self):
        number_of_tokens = 3
        example_input_words_1 = torch.LongTensor([1,2,3])
        example_input_words_2 = torch.LongTensor([[1,2,3], [2,3,4]])

        token_encoder = TokenEncoder(self.d_model, self.vocab)

        input_embeddings_1 = token_encoder.forward(example_input_words_1)
        input_embeddings_2 = token_encoder.forward(example_input_words_2)

    def test_transformer_encoder_with_word_inputs(self):
        number_of_tokens = 3
        number_of_encoder_layers = 6
        example_input_words_1 = torch.LongTensor([1,2,3])
        example_input_words_2 = torch.LongTensor([[1,2,3], [2,3,4]])

        token_encoder = TokenEncoder(self.d_model, self.vocab)
        encoder_layer = TransformerEncoderLayer(self.d_model, self.number_of_heads)
        encoder = TransformerEncoder(encoder_layer, number_of_encoder_layers)

        input_embeddings_1 = token_encoder.forward(example_input_words_1)
        input_embeddings_2 = token_encoder.forward(example_input_words_2)

        memory_1 = encoder.forward(input_embeddings_1)
        memory_2 = encoder.forward(input_embeddings_2)

    def test_transformer_model(self):
        number_of_encoder_layers = 6
        number_of_decoder_layers = 6
        example_input_words_1 = torch.LongTensor([1, 2, 3])
        example_input_words_2 = torch.LongTensor([[1, 2, 3], [2, 3, 4]])
        example_target_words_1 = torch.LongTensor([2, 3, 4])
        example_target_words_2 = torch.LongTensor([[2, 3, 4], [3, 4, 5]])

        transformer_model = TransformerModel(number_of_encoder_layers, number_of_decoder_layers, self.d_model,
                                             self.vocab, self.number_of_heads)

        output_1 = transformer_model.forward(example_input_words_1, example_target_words_1)
        output_2 = transformer_model.forward(example_input_words_2, example_target_words_2)