import pytest
import torch
import torch.nn.functional as F
from src.conv import conv2d

from .helpers import EPSILON

# I'm not padding images myself, so padding is 0 and stride 1 to be safe
PADDING = 0
STRIDE = 1


@pytest.mark.parametrize(
    "input_size, kernel_size",
    [
        ((2, 3, 27, 27), (20, 3, 3, 3)),
        ((1, 1, 10, 10), (5, 1, 3, 3)),
        ((4, 3, 32, 32), (10, 3, 5, 5)),
    ],
)
def test_conv(input_size, kernel_size):
    # Generate random input tensor
    input_batch = torch.rand(*input_size)
    kernel = torch.rand(*kernel_size)

    output = conv2d(input_batch, kernel, padding=PADDING, stride=STRIDE)
    assert torch.allclose(
        output,
        F.conv2d(input_batch, kernel, padding=PADDING, stride=STRIDE),
        atol=EPSILON,
    ), f"Mismatch in outputs for input size {input_size}, kernel size {kernel_size}, stride {STRIDE}, and padding {PADDING}."
