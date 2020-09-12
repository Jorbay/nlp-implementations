from .attention import MultiHeadAttention
from .embedding import Embedding
from .toolbox import SingleLayerPerceptron, make_mask
import torch

class TestClass:

    def test_scaledDotProductAttention(self):
        embeds = Embedding(512, 32000)
        d_model = 512
        d_k = 64
        number_of_heads = int(d_model / d_k)
        number_of_tokens = 1

        example_word = torch.tensor([int(1)], dtype=torch.long)
        example_embedded= embeds(example_word)

        slp = SingleLayerPerceptron(d_model, d_k)
        example_key = slp.forward(example_embedded)
        example_value = slp.forward(example_embedded)
        example_query = slp.forward(example_embedded)

        mask = make_mask(number_of_tokens)
        print("Mask size is: ")
        print(mask.size())

        multi_head_attention = MultiHeadAttention(number_of_heads, d_model)

        multi_head_attention.scaledDotProductAttention(example_query, example_key, example_value)
        multi_head_attention.scaledDotProductAttention(example_query, example_key, example_value, mask)