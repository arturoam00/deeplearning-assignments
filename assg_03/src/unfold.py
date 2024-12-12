import torch

from .utils import SizeTuple, out


def unfold(
    input_batch: torch.Tensor, kernel_size: SizeTuple, padding: int, stride: int
):
    """
    p = output_width * output_width
    k = kernel_size * kernel_size * input_channels"""
    b, c, h, w = input_batch.size()
    _, _c, hk, wk = kernel_size

    assert c == _c, "Input channels must match"

    k = hk * wk * c  # kernel params
    h_out, w_out = out(h, hk, padding=padding, stride=stride), out(
        w, wk, padding=padding, stride=stride
    )
    p = h_out * w_out  # number of image patches

    output = torch.zeros((b, p, k))

    for batch in range(b):
        # iterate over rows in the output
        # each row will be a flattened image patch
        for row in range(p):
            # select patch
            i = row // w_out
            j = row % (w - wk + 1)
            start_row, start_col = i * stride, j * stride
            patch = input_batch[
                batch, :, start_row : start_row + hk, start_col : start_col + wk
            ]

            assert patch.size() == (
                c,
                hk,
                wk,
            ), f"Patch of size {patch.size()} invalid. Calculated with indices {i, j}, row = {row}"

            # I can already flatten it this way, but we speaking forloops
            assert (
                len(patch.flatten()) == k
            ), f"Flattened patch lenght is: {len(patch.flatten())}. Expected: {k}"

            # flatten patch
            counter = 0
            for ch_patch in range(c):
                for row_patch in range(hk):
                    for col_patch in range(wk):
                        output[batch, row, counter] = patch[
                            ch_patch, row_patch, col_patch
                        ]
                        counter += 1
    # This way the output will have size (batch_size, p, k)
    return output
