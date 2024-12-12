from typing import Tuple

import torch
import torch.nn.functional as F

from .utils import Conv2DParams


class Conv2DFunc(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward
    passes which operate on Tensors.
    """

    @staticmethod
    def forward(
        ctx,
        input_batch: torch.Tensor,
        kernel: torch.Tensor,
        padding: int = 1,
        stride: int = 1,
    ) -> torch.Tensor:
        """
        In the forward pass we receive a Tensor containing the input
        and return a Tensor containing the output. ctx is a context
        object that can be used to stash information for backward
        computation. You can cache arbitrary objects for use in the
        backward pass using the ctx.save_for_backward method.
        """

        # store objects for the backward
        ctx.save_for_backward(input_batch, kernel)
        ctx.padding = padding
        ctx.stride = stride

        params = Conv2DParams(
            kernel_size=kernel.size(),
            input_size=input_batch.size(),
            padding=padding,
            stride=stride,
        )

        unfolded = F.unfold(  # U
            input_batch,
            (params.hk, params.wk),
            padding=params.padding,
            stride=params.stride,
        ).transpose(
            1, 2
        )  # \tilde{U}

        assert unfolded.size() == (
            params.b,
            params.p,
            params.k,
        ), f"Expected: {(params.b, params.p, params.k)}, but got: {unfolded.size()}"

        output_batch = unfolded.matmul(
            kernel.reshape(params.c_out, -1).t()  # \tilde{U} @ \tilde{W} = Y'
        ).transpose(
            1, 2
        )  # This op together with the next reshape form Y

        assert output_batch.size() == (params.b, params.c_out, params.p)

        return output_batch.view(
            params.b, params.c_out, params.h_out, params.w_out
        )  # Y

    @staticmethod
    def backward(ctx, grad_output) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        In the backward pass we receive a Tensor containing the
        gradient of the loss with respect to the output, and we need
        to compute the gradient of the loss with respect to the
        input
        """
        # retrieve stored objects
        (input_batch, kernel) = ctx.saved_tensors
        padding = ctx.padding
        stride = ctx.stride
        # your code here

        params = Conv2DParams(
            kernel_size=kernel.size(),
            input_size=input_batch.size(),
            padding=padding,
            stride=stride,
        )

        grad_output = grad_output.view(
            params.b, params.c_out, -1
        )  # shape: (b, c_out, p)

        assert grad_output.size() == (
            params.b,
            params.c_out,
            params.p,
        ), f"Expected: {(params.b, params.c_out, params.p)}, got: {grad_output.size()}"

        # Gradient w.r.t. kernel
        unfolded = F.unfold(
            input_batch,
            kernel_size=(params.hk, params.wk),
            padding=params.padding,
            stride=params.stride,
        ).transpose(1, 2)

        assert unfolded.size() == (
            params.b,
            params.p,
            params.k,
        ), f"Expected: {(params.b, params.p, params.k)}, got: {unfolded.size()}"

        grad_kernel = (
            grad_output.transpose(0, 1)
            .reshape(params.c_out, -1)
            .matmul(unfolded.reshape(params.b * params.p, params.k))
        )

        assert grad_kernel.size() == (
            params.c_out,
            params.k,
        ), f"Expected: {(params.c_out, params.k)}, got: {grad_kernel.size()}"

        grad_kernel = grad_kernel.view(params.c_out, params.c, params.hk, params.wk)

        # Gradient w.r.t. input
        assert grad_output.size() == (
            params.b,
            params.c_out,
            params.p,
        ), f"Expected: {(params.b, params.c_out, params.p)}, got: {grad_output.size()}"

        grad_input_unfolded = (
            kernel.view(params.c_out, -1)
            .t()  # this is \tilde{W}, shape: (k, c_out)
            .matmul(grad_output.transpose(0, 1).reshape(params.c_out, -1))
        ).reshape(params.k, params.b, params.p)

        assert grad_input_unfolded.size() == (
            params.k,
            params.b,
            params.p,
        ), f"Expected: {(params.k, params.b, params.p)}, got: {grad_input_unfolded.size()}"

        grad_input = F.fold(
            grad_input_unfolded.transpose(0, 1),
            output_size=(params.h, params.w),
            kernel_size=(params.hk, params.wk),
            padding=params.padding,
            stride=params.stride,
        )

        assert grad_input.size() == (
            params.b,
            params.c,
            params.h,
            params.w,
        ), f"Expected: {(params.b, params.c, params.h, params.w)}, got: {grad_input.size()}"

        return (
            grad_input,
            grad_kernel,
            None,
            None,
        )
