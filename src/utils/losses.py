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
    mu_k: torch.Tensor,  # [batch_size, k, seq_len, n_features]
    var_k: torch.Tensor,  # [batch_size, k, seq_len, n_features]
    log_w_k: torch.Tensor,  # [batch_size, k]
) -> torch.Tensor:
    """
    Compute KL divergence between a diagonal Gaussian posterior and a mixture of diagonal
    Gaussians prior for masked features only.

    KL(q(z|x) || p(z)) = log q(z|x) - log p(z)
    where p(z) = sum_i w_i * N(z; mu_k[i], var_k[i])
    """
    # Compute log q(z|x) - posterior log-density for masked features only
    log_q_z_x = gaussian_log_likelihood(z, mu_x, var_x, feature_mask, (1, 2))

    # Compute log p(z) - mixture prior log-density for masked features only
    # Reshape tensors for broadcasting
    z_expanded = z.unsqueeze(1)  # [batch_size, 1, seq_len, n_features]
    feature_mask_expanded = feature_mask.unsqueeze(
        1
    )  # [batch_size, 1, seq_len, n_features]

    # mu_k and var_k are already [batch_size, k, seq_len, n_features]
    # Compute log-likelihood for each mixture component
    log_component_densities = gaussian_log_likelihood(
        z_expanded, mu_k, var_k, feature_mask_expanded, (2, 3)
    )  # [batch_size, k]

    # Use learnable mixture weights
    # Compute log p(z) = log(sum_i w_i * p_i(z)) = logsumexp(log(w_i) + log(p_i(z)))
    log_p_z = torch.logsumexp(log_w_k + log_component_densities, dim=1)  # [batch_size]

    kl_divergence = log_q_z_x - log_p_z
    return kl_divergence
