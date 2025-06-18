"""
Custom loss functions.
All loss functions return output of shape [batch_size]
"""

import torch
from typing import Tuple, Optional


def gaussian_log_likelihood(
    x: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    feature_mask: torch.Tensor,
    masked_dims: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Compute the Gaussian log-likelihood for masked features.
    Log-likelihood = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
    """
    if masked_dims is None:
        # Default to summing over all dimensions except batch dimension
        masked_dims = tuple(range(1, x.ndim))
    # Compute the Gaussian log-likelihood
    log_likelihood = -0.5 * torch.log(2 * torch.pi * var) - 0.5 * (x - mu) ** 2 / var
    masked_log_likelihood = log_likelihood * feature_mask
    return torch.sum(masked_log_likelihood, dim=masked_dims)


def compute_gaussian_kl_divergence(
    feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features]
    mu_x: torch.Tensor,  # [batch_size, seq_len, n_features]
    var_x: torch.Tensor,  # [batch_size, seq_len, n_features]
    mu_p: torch.Tensor,  # [1, n_mixtures, n_features]
    var_p: torch.Tensor,  # [1, n_mixtures, n_features]
) -> torch.Tensor:
    """
    Compute KL divergence between two diagonal Gaussians for masked features only.
    KL(q(z|x) || p(z)) = 0.5 * [log(var_p/var_x) + var_x/var_p + (mu_x - mu_p)^2/var_p - 1]
    """
    kl_per_dim = 0.5 * (
        torch.log(var_p / var_x) + var_x / var_p + (mu_x - mu_p) ** 2 / var_p - 1.0
    )
    kl_masked = kl_per_dim * feature_mask
    # sum over masked dims
    kl_divergence = torch.sum(kl_masked, dim=(1, 2))
    return kl_divergence


def compute_mixture_kl_divergence(
    z: torch.Tensor,  # [batch_size, seq_len, n_features]
    feature_mask: torch.Tensor,  # [batch_size, seq_len, n_features]
    mu_x: torch.Tensor,  # [batch_size, seq_len, n_features]
    var_x: torch.Tensor,  # [batch_size, seq_len, n_features]
    mu_k: torch.Tensor,  # [k, seq_len, n_features]
    var_k: torch.Tensor,  # [k, seq_len, n_features]
) -> torch.Tensor:
    """
    Compute KL divergence between a diagonal Gaussian posterior and a mixture of diagonal
    Gaussians prior for masked features only.

    KL(q(z|x) || p(z)) = log q(z|x) - log p(z)
    where p(z) = (1/k) * sum_i N(z; mu_k[i], var_k[i])
    """
    # Compute log q(z|x) - posterior log-density for masked features only
    log_q_z_x = gaussian_log_likelihood(z, mu_x, var_x, feature_mask, (1, 2))

    # Compute log p(z) - mixture prior log-density for masked features only
    # Reshape tensors for broadcasting
    z_expanded = z.unsqueeze(0)  # [1, batch_size, seq_len, n_features]
    mu_k_expanded = mu_k.unsqueeze(1)  # [k, 1, seq_len, n_features]
    var_k_expanded = var_k.unsqueeze(1)  # [k, 1, seq_len, n_features]
    feature_mask_expanded = feature_mask.unsqueeze(
        0
    )  # [1, batch_size, seq_len, n_features]

    # Compute log-likelihood for each mixture component
    log_component_densities = gaussian_log_likelihood(
        z_expanded, mu_k_expanded, var_k_expanded, feature_mask_expanded, (2, 3)
    )

    # Add uniform mixture weights: log(1/k) = -log(k)
    k = mu_k.shape[0]
    log_mixture_weights = -torch.log(
        torch.tensor(k, dtype=torch.float32, device=z.device)
    )

    # Compute log p(z) = log(sum_i (1/k) * p_i(z)) = logsumexp(log(1/k) + log(p_i(z)))
    log_p_z = torch.logsumexp(
        log_mixture_weights + log_component_densities, dim=0
    )  # [batch_size]

    kl_divergence = log_q_z_x - log_p_z
    kl_clamped = torch.clamp(kl_divergence, min=0.0)
    return kl_clamped
