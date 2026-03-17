"""
Sub-contribution 1.2: Gaussian Distance Metric on the Product Manifold
=======================================================================

Defines d_G(G₁, G₂) — the geometrically proper distance between two
3D Gaussian primitives, respecting the structure of each parameter space:

    d_G(G₁, G₂)² = ||μ₁ - μ₂||²                              (position: Euclidean)
                  + β₁ · ||log(Σ₁) - log(Σ₂)||²_F            (covariance: Log-Euclidean)
                  + β₂ · ||c₁ - c₂||²                          (color: Euclidean)
                  + β₃ · (α₁ - α₂)²                            (opacity: Euclidean)

Key insight: The covariance distance uses the Log-Euclidean metric, NOT
the naive Frobenius norm. This is critical because:
  1. Σ lives on the SPD manifold S⁺₃, not in Euclidean space
  2. The Log-Euclidean metric makes S⁺₃ a flat (geodesically complete) space
  3. This makes the Gaussian component of our consensus CONVEX (Sub-contribution 1.3)

References:
- Log-Euclidean metrics on SPD manifolds: Arsigny et al., SIAM J. Matrix Anal. 2007
- Product manifold optimization: Boumal, Cambridge University Press, 2023
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def log_euclidean_distance(
    cov1: torch.Tensor,  # [N, 3, 3] or [N, 6] (upper tri)
    cov2: torch.Tensor,  # [N, 3, 3] or [N, 6]
) -> torch.Tensor:
    """
    Log-Euclidean distance between covariance matrices.
    
    d_LE(Σ₁, Σ₂) = ||log(Σ₁) - log(Σ₂)||_F
    
    where log is the matrix logarithm (eigendecomposition-based).
    
    For 3×3 SPD matrices, this is computed via:
        1. Eigendecompose: Σ = Q Λ Q^T
        2. log(Σ) = Q log(Λ) Q^T  (apply log to eigenvalues)
        3. Frobenius norm of the difference
    
    Args:
        cov1, cov2: Covariance matrices [N, 3, 3] or upper-tri [N, 6]
    Returns:
        distances: [N] — scalar distance per pair
    """
    # Convert upper triangular to full 3x3 if needed
    if cov1.dim() == 2 and cov1.shape[-1] == 6:
        cov1 = upper_tri_to_full(cov1)
        cov2 = upper_tri_to_full(cov2)
    
    # Eigendecomposition (batched)
    eigvals1, eigvecs1 = torch.linalg.eigh(cov1)  # [N, 3], [N, 3, 3]
    eigvals2, eigvecs2 = torch.linalg.eigh(cov2)
    
    # Clamp eigenvalues for numerical stability (must be positive for log)
    eigvals1 = eigvals1.clamp(min=1e-8)
    eigvals2 = eigvals2.clamp(min=1e-8)
    
    # Matrix logarithm: log(Σ) = Q diag(log(λ)) Q^T
    log_cov1 = eigvecs1 @ torch.diag_embed(eigvals1.log()) @ eigvecs1.transpose(-1, -2)
    log_cov2 = eigvecs2 @ torch.diag_embed(eigvals2.log()) @ eigvecs2.transpose(-1, -2)
    
    # Frobenius norm of difference
    diff = log_cov1 - log_cov2
    dist = torch.sqrt((diff ** 2).sum(dim=(-1, -2)))  # [N]
    
    return dist


def upper_tri_to_full(upper_tri: torch.Tensor) -> torch.Tensor:
    """Convert [N, 6] upper-triangular representation to [N, 3, 3] symmetric matrix."""
    N = upper_tri.shape[0]
    full = torch.zeros(N, 3, 3, device=upper_tri.device, dtype=upper_tri.dtype)
    
    # Upper triangle indices for 3x3
    idx = torch.triu_indices(3, 3)
    full[:, idx[0], idx[1]] = upper_tri
    full[:, idx[1], idx[0]] = upper_tri  # Symmetric
    
    return full


class GaussianDistanceMetric:
    """
    Computes the distance d_G between pairs of 3D Gaussian primitives.
    
    This is the core metric used in the DUAG-C consensus (Eq. 2).
    The weighting coefficients β₁, β₂, β₃ control the relative importance
    of covariance, color, and opacity alignment.
    
    Default values are chosen so that each component contributes roughly
    equally when Gaussians are at a "typical" distance from each other.
    """
    
    def __init__(
        self,
        beta_cov: float = 0.1,     # β₁: covariance weight
        beta_color: float = 1.0,   # β₂: color weight  
        beta_opacity: float = 0.5, # β₃: opacity weight
    ):
        self.beta_cov = beta_cov
        self.beta_color = beta_color
        self.beta_opacity = beta_opacity
    
    def compute(
        self,
        positions_1: torch.Tensor,    # [M, 3]
        covariances_1: torch.Tensor,  # [M, 6] upper tri
        colors_1: torch.Tensor,       # [M, 3]
        opacities_1: torch.Tensor,    # [M, 1]
        positions_2: torch.Tensor,    # [M, 3]
        covariances_2: torch.Tensor,  # [M, 6]
        colors_2: torch.Tensor,       # [M, 3]
        opacities_2: torch.Tensor,    # [M, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute d_G² between M pairs of Gaussians.
        
        Returns:
            total_distance: [M] — total d_G² per pair
            component_distances: [M, 4] — per-component (pos, cov, color, opacity)
        """
        # Position: Euclidean
        d_pos = ((positions_1 - positions_2) ** 2).sum(dim=-1)  # [M]
        
        # Covariance: Log-Euclidean
        d_cov = log_euclidean_distance(covariances_1, covariances_2) ** 2  # [M]
        
        # Color: Euclidean
        d_color = ((colors_1 - colors_2) ** 2).sum(dim=-1)  # [M]
        
        # Opacity: Euclidean
        d_opacity = ((opacities_1 - opacities_2) ** 2).squeeze(-1)  # [M]
        
        # Weighted sum
        total = d_pos + self.beta_cov * d_cov + self.beta_color * d_color + self.beta_opacity * d_opacity
        
        components = torch.stack([d_pos, d_cov, d_color, d_opacity], dim=-1)  # [M, 4]
        
        return total, components
    
    def compute_weighted(
        self,
        positions_1: torch.Tensor, covariances_1: torch.Tensor,
        colors_1: torch.Tensor, opacities_1: torch.Tensor,
        positions_2: torch.Tensor, covariances_2: torch.Tensor,
        colors_2: torch.Tensor, opacities_2: torch.Tensor,
        weights: torch.Tensor,  # [M] — uncertainty-based weights w_k = 1/σ_k²
    ) -> torch.Tensor:
        """
        Compute the weighted consensus objective (Eq. 2):
            Σ_k  w_k · d_G(G_k^i, G_k^j)²
        
        Returns:
            scalar — the total weighted distance (to be minimized)
        """
        total, _ = self.compute(
            positions_1, covariances_1, colors_1, opacities_1,
            positions_2, covariances_2, colors_2, opacities_2,
        )
        return (weights * total).sum()
