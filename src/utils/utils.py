import torch
from typing import Union


def center_trim(tensor: torch.Tensor, reference: Union[torch.Tensor, int]) -> torch.Tensor:
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.

    Args:

        tensor (torch.Tensor): Tensor to be center trimmed.
        reference (torch.Tensor or int): Reference tensor or length.

    Returns:

        torch.Tensor: Center trimmed tensor.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor
