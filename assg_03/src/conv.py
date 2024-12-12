import torch

from .utils import out


def conv2d(input_batch: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int):
    b, c, h, w = input_batch.size()
    c_out, _c, hk, wk = kernel.size()
    h_out, w_out = out(h, hk, padding=padding, stride=stride), out(
        w, wk, padding=padding, stride=stride
    )

    assert c == _c, "Input channels must match"
    output = torch.zeros((b, c_out, h_out, w_out))

    for batch in range(b):
        for ch in range(c_out):
            for row in range(h_out):
                for col in range(w_out):
                    start_row = row * stride
                    start_col = col * stride
                    output[batch, ch, row, col] = (
                        input_batch[
                            batch,
                            :,
                            start_row : start_row + hk,
                            start_col : start_col + wk,
                        ]
                        * kernel[ch, :, :, :]
                    ).sum()
    return output
