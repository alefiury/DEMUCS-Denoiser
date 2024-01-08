import torch
import auraloss
import torch.nn.functional as F


def loss_function(estimate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Loss function for training the model.

    Args:
        estimate (torch.Tensor): Estimated source.
        source (torch.Tensor): Ground truth source.

    Returns:

        torch.Tensor: Total loss.
    """
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(w_sc=0.5, w_log_mag=0.5)
    mrstft_loss = mrstft(estimate, source)
    l1_loss = F.l1_loss(source, estimate)
    total_loss = mrstft_loss + l1_loss
    return total_loss, mrstft_loss, l1_loss