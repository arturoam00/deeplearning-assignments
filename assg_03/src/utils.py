import warnings
from typing import Tuple

SizeTuple = Tuple[int, int, int, int]


def out(x: int, ks: int, padding: int, stride: int) -> int:
    """For a given `x` (width or height), calculate the size
    (width or height) of the output, this is, number of images
    patches per row/column
    """
    res = (x - ks + 2 * padding) / stride + 1
    if not res.is_integer():
        warnings.warn(
            f"Result from num of patches op. is not integer ({res}), skipping pixels. Don't panic, we can go on :)"
        )
        return (x - ks + 2 * padding) // stride + 1
    return int(res)


class InvalidSizeTuple(Exception):
    pass


class Conv2DParams:
    """This class stores all parameters and provides
    some useful computations for convolutions

    b -> batches
    h -> height
    w -> width
    c -> channels

    When no sufix is set, assume input, when output, sufix `_out` is set.
    For kernel width and height: wk, hk respectively
    """

    def __init__(
        self,
        kernel_size: SizeTuple,
        input_size: SizeTuple,
        padding: int = 1,
        stride: int = 1,
    ):
        # Kernel related params
        self.c_out, self.c, self.hk, self.wk = self._check_size_tuple(kernel_size)
        self.k = self.hk * self.wk * self.c  # Total number of values per patch

        # Input related params
        self.b, _c, self.h, self.w = self._check_size_tuple(input_size)
        assert _c == self.c, f"Mismatch of input channels in input and kernel"

        # Output related params
        self.h_out = out(self.h, self.hk, padding=padding, stride=stride)
        self.w_out = out(self.w, self.wk, padding=padding, stride=stride)
        self.p = self.h_out * self.w_out  # total number of patches

        self.padding = padding
        self.stride = stride

    def _check_size_tuple(self, size) -> SizeTuple:
        if (
            not isinstance(size, tuple)
            or not len(size) == 4
            or not all(isinstance(e, int) for e in size)
        ):
            raise InvalidSizeTuple(f"Size tuple: {size} is not valid")
        return size
