import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import Conv2DParams, SizeTuple


class Conv2DModule(nn.Module):

    def __init__(
        self, kernel_size: SizeTuple, stride: int = 1, padding: int = 1
    ) -> None:
        super().__init__()
        # Pytorch's kernel would have shape (out_channles, in_channels, *kernel_size)
        # Ours is built to be a simple 2d matrix, but is equivalent if reshaped
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = torch.randn(
            math.prod(kernel_size[1:]), kernel_size[0]
        )  # shape: (k, c_out)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        # parameters
        params = Conv2DParams(
            kernel_size=self.kernel_size,
            input_size=input_batch.size(),
            padding=self.padding,
            stride=self.stride,
        )

        unfolded = F.unfold(
            input_batch,
            kernel_size=(params.hk, params.wk),
            padding=params.padding,
            stride=params.stride,
            # We transpose, because Pytorch returns flattened patches as columns
            # I want them as rows
        ).transpose(1, 2)

        assert unfolded.size() == (
            params.b,
            params.p,
            params.k,
        ), f"Expected: {(params.b, params.p, params.k)} but got: {unfolded.size()}"

        # First reshape, then Y = XW, then reshape back
        output = (
            unfolded.reshape(params.b * params.p, params.k)
            .matmul(self.kernel)
            .reshape(params.b, params.p, params.c_out)  # This is actually Y
            .transpose(1, 2)  # This is needed for later
        )

        assert output.size() == (
            params.b,
            params.c_out,
            params.p,
        ), f"Expected: {(params.b, params.c_out, params.p)} but got: {output.size()}"

        return output.reshape(params.b, params.c_out, params.h_out, params.w_out)
