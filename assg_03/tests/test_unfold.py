import pytest
import torch
from src.unfold import unfold

from .helpers import EPSILON

# I'm not padding images myself, so padding is 0 and stride 1 to be safe
PADDING = 0
STRIDE = 1


@pytest.mark.parametrize(
    "input_size, kernel_size",
    [
        ((2, 3, 10, 10), (20, 3, 3, 3)),
        ((1, 3, 8, 8), (10, 3, 2, 2)),
        ((4, 1, 15, 15), (5, 1, 5, 5)),
    ],
)
def test_unfold(input_size, kernel_size):

    input_batch = torch.rand(*input_size)
    hk, wk = kernel_size[2:]

    # Apply the custom unfold function
    unfolded = unfold(
        input_batch=input_batch, kernel_size=kernel_size, padding=PADDING, stride=STRIDE
    )

    # Use PyTorch's unfold for comparison
    pytorch_unfolded = torch.nn.functional.unfold(
        input_batch, (hk, wk), padding=PADDING, stride=STRIDE
    )
    pytorch_unfolded = pytorch_unfolded.permute(
        0, 2, 1
    )  # Pytorch unfold returns (b, k, p)

    # Compare results
    assert torch.allclose(
        unfolded, pytorch_unfolded, atol=EPSILON
    ), "Pytorch says something else"
