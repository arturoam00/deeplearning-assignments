import pytest
import torch
import torch.nn.functional as F
from src.conv_layer_module import Conv2DModule

from .helpers import EPSILON


@pytest.mark.parametrize("batch_size", [2, 1])
@pytest.mark.parametrize("in_channels", [3, 5])
@pytest.mark.parametrize("out_channels", [4, 10])
@pytest.mark.parametrize("height", [10, 5])
@pytest.mark.parametrize("width", [10, 5])
@pytest.mark.parametrize("kernel_size", [5, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_module(
    batch_size, in_channels, out_channels, height, width, kernel_size, padding, stride
):
    input_batch = torch.randn(batch_size, in_channels, height, width)
    # me
    conv = Conv2DModule(
        kernel_size=(out_channels, in_channels, kernel_size, kernel_size),
        stride=stride,
        padding=padding,
    )
    output_batch = conv(input_batch)

    # Pytorch needs a differently shaped kernel ...
    kernel_tensor = conv.kernel.transpose(0, 1).reshape(
        out_channels, in_channels, kernel_size, kernel_size
    )
    output_torch = F.conv2d(input_batch, kernel_tensor, stride=stride, padding=padding)

    assert (
        output_torch - output_batch
    ).abs().max() < EPSILON, "Conv2D module output mismatch!"
