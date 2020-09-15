from .attention import MultiHeadAttention
from .embedding import Embedding
from .toolbox import SingleLayerPerceptron, make_mask
import torch

class TestClass:

    def test_scaledDotProductAttention(self):
        d_model = 512
        d_k = 64
        number_of_heads = int(d_model / d_k)
        number_of_tokens = 3

        example_query = torch.tensor([[1, 1], [1, 1], [1,1]], dtype=torch.float64)
        example_key = example_query.detach().clone()
        example_value = example_query.detach().clone()
        mask = make_mask(number_of_tokens)

        multi_head_attention = MultiHeadAttention(number_of_heads, d_model)

        result_without_mask = multi_head_attention.scaledDotProductAttention(example_query, example_key, example_value)
        result_with_mask = multi_head_attention.scaledDotProductAttention(example_query, example_key, example_value,
                                                                          mask)
        expected_result_without_mask = example_query.detach().clone()
        expected_result_with_mask = torch.tensor([[1/3, 1/3], [5/6, 5/6], [11/6, 11/6]], dtype=torch.float64)

        assert torch.equal(result_without_mask, expected_result_without_mask)

        #TODO: Fix this assert for the expected result of mask.
        #assert torch.equal(result_with_mask, expected_result_with_mask)