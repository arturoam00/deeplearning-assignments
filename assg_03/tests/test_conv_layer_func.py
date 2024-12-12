import pytest
import torch
import torch.nn.functional as F
from src.conv_layer_func import Conv2DFunc


@pytest.mark.parametrize("batch_size", [2, 1])
@pytest.mark.parametrize("in_channels", [3, 5])
@pytest.mark.parametrize("out_channels", [4, 10])
@pytest.mark.parametrize("height", [10, 5])
@pytest.mark.parametrize("width", [10, 5])
@pytest.mark.parametrize("kernel_size", [5, 3])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv2d_function(
    batch_size, in_channels, out_channels, height, width, kernel_size, padding, stride
):
    """
    Tests both the forward and backward pass of the custom Conv2D layer.
    """

    # Input and kernel tensors
    input_tensor = torch.randn(
        batch_size, in_channels, height, width, requires_grad=True, dtype=torch.double
    )
    kernel_tensor = torch.randn(
        out_channels,
        in_channels,
        kernel_size,
        kernel_size,
        requires_grad=True,
        dtype=torch.double,
    )
    C2D = Conv2DFunc.apply

    # Forward pass (Custom vs PyTorch)
    output_custom = C2D(input_tensor, kernel_tensor, padding, stride)

    output_torch = F.conv2d(input_tensor, kernel_tensor, padding=padding, stride=stride)
    assert torch.allclose(
        output_custom, output_torch, atol=1e-6
    ), "Forward pass mismatch"

    # Backward pass
    # Gradient check
    gradcheck_input = (
        input_tensor.clone().detach().requires_grad_(True),
        kernel_tensor.clone().detach().requires_grad_(True),
        padding,
        stride,
    )
    assert torch.autograd.gradcheck(
        C2D, gradcheck_input, eps=1e-6, atol=1e-4
    ), "Gradcheck failed"
