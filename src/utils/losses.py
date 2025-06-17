import torch
from typing import Tuple


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Tuple[int, ...]):
    """Mean over `dim`, ignoring False in `mask`."""
    masked = tensor * mask
    return masked.sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1))


def compute_gaussian_kl_divergence(
    mu_x: torch.Tensor,
    var_x: torch.Tensor,
    mu_p: torch.Tensor,
    var_p: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between two diagonal Gaussians for masked features only.
    KL(q(z|x) || p(z)) = 0.5 * [log(var_p/var_x) + var_x/var_p + (mu_x - mu_p)^2/var_p - 1]
    """
    kl_per_dim = 0.5 * (
        torch.log(var_p / var_x) + var_x / var_p + (mu_x - mu_p) ** 2 / var_p - 1.0
    )
    kl_masked = kl_per_dim * feature_mask
    kl_divergence = masked_mean(kl_masked, feature_mask, dim=(1, 2)).mean()
    return kl_divergence


def compute_mixture_kl_divergence(
    z: torch.Tensor,
    mu_x: torch.Tensor,
    var_x: torch.Tensor,
    mu_k: torch.Tensor,
    var_k: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between a diagonal Gaussian posterior and a mixture of diagonal
    Gaussians prior for masked features only.

    KL(q(z|x) || p(z)) = log q(z|x) - log p(z)
    """
    # Compute log q(z|x) - posterior log-density for masked features only
    log_q_z_x_all = -0.5 * (torch.log(var_x) + (z - mu_x) ** 2 / var_x)
    log_q_z_x_masked = log_q_z_x_all * feature_mask
    log_q_z_x = torch.sum(log_q_z_x_masked, dim=(1, 2))

    # Compute log p(z) - mixture prior log-density for masked features only
    z_expanded = z.unsqueeze(0)
    mu_k_expanded = mu_k.unsqueeze(1)
    var_k_expanded = var_k.unsqueeze(1)

    log_component_densities_all = -0.5 * (
        torch.log(var_k_expanded) + (z_expanded - mu_k_expanded) ** 2 / var_k_expanded
    )
    log_component_densities_masked = (
        log_component_densities_all * feature_mask.unsqueeze(0)
    )
    log_component_densities = torch.sum(log_component_densities_masked, dim=(2, 3))

    log_p_z = torch.logsumexp(log_component_densities, dim=0)
    kl_divergence = (log_q_z_x - log_p_z).mean()
    return kl_divergence


def gaussian_nll_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the Gaussian Negative Log-Likelihood loss for masked features.
    """
    gaussian_nll = 0.5 * torch.log(var) + 0.5 * (x - mu) ** 2 / var
    masked_gaussian_nll = gaussian_nll * feature_mask
    return masked_mean(masked_gaussian_nll, feature_mask, dim=(1, 2)).mean()
