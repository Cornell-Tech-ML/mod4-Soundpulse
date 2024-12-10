from typing import Optional, Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Computes the maximum of the input tensor along the specified dimension."""
        max_result = a.f.max_reduce(a, int(dim.item()))
        mask = a == max_result
        ctx.save_for_backward(mask, dim)
        return max_result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes the gradient for the sum operation."""
        mask, dim = ctx.saved_values

        grad_input = (mask * grad_output) / mask.sum(int(dim.item()))
        return grad_input, 0.0


def max(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the max along the specified dimension."""
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Returns the indices of the maximum values along an axis."""
    max_result = max(input, dim=dim)
    return input == max_result


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    input = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)
    output = input.contiguous().view(batch, channel, new_height, new_width, kh * kw)

    return output, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Implementation of Average Pooling in 2D."""
    batch, channel, _, _ = input.shape
    tiled_input, new_height, new_width = tile(input, kernel)
    # Compute the average over the last dimension
    result = tiled_input.mean(dim=len(tiled_input.shape) - 1)
    return result.view(batch, channel, new_height, new_width)


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Implementation of Max Pooling in 2D."""
    batch, channel, _, _ = input.shape
    tiled_input, new_height, new_width = tile(input, kernel)
    # Compute the max over the last dimension
    result = max(tiled_input, dim=len(tiled_input.shape) - 1)
    return result.view(batch, channel, new_height, new_width)


def softmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Applies the softmax function to the input tensor."""
    if dim is None:
        input = input.contiguous().view(input.size)
        dim = 0

    exp_result = input.exp()
    exp_sum = exp_result.sum(dim=dim)

    return exp_result / exp_sum


def logsoftmax(input: Tensor, dim: Optional[int] = None) -> Tensor:
    """Applies the log softmax function to the input tensor."""
    max_i = max(input, dim)
    stable_input = input - max_i
    return stable_input - stable_input.exp().sum(dim=dim).log()


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Applies dropout to the input tensor for regularization."""
    if ignore:
        return input

    if p >= 1.0:
        return input.zeros()

    return input * (rand(input.shape) > p)
