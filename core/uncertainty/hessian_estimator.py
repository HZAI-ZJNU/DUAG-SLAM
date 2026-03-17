"""
Sub-contribution 1.1: Epistemic Uncertainty Estimation for 3D Gaussian Splatting
=================================================================================

Scientific basis: Laplace approximation of the posterior over Gaussian parameters.

    p(θ_k | D) ≈ N(θ_k*, H_k⁻¹)

where H_k is the Hessian of the rendering loss w.r.t. Gaussian k's parameters.
The uncertainty is:

    σ_k² = Tr(H_k⁻¹)     ... (Eq. 5 in thesis)

We use the diagonal Gauss-Newton approximation for efficiency:

    H_k ≈ diag(J_k^T J_k)

where J_k is the Jacobian of the rendered pixel values w.r.t. θ_k.

This module hooks into the backward pass of the Gaussian rasterizer
to accumulate the diagonal Hessian without additional forward passes.

References:
- Laplace approximation: MacKay, 1992
- Diagonal GGN for neural nets: Martens, 2014
- 3DGS rendering: Kerbl et al., ACM TOG 2023
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class EpistemicUncertaintyEstimator:
    """
    Computes per-Gaussian epistemic uncertainty from the rendering loss Hessian.
    
    Usage:
        estimator = EpistemicUncertaintyEstimator(num_params_per_gaussian=14)
        
        # During training/mapping loop:
        estimator.accumulate(loss, gaussian_params)  # Call after each frame
        
        # Every N frames:
        uncertainties = estimator.compute()  # Returns σ_k² for each Gaussian
        estimator.reset()  # Reset accumulator for next window
    """
    
    def __init__(
        self,
        num_params_per_gaussian: int = 14,  # 3(pos) + 6(cov) + 3(color) + 1(opacity) + 1(SH)
        accumulation_window: int = 10,       # Compute uncertainty every N frames
        damping: float = 1e-6,               # Numerical stability
        device: str = "cuda",
    ):
        self.num_params = num_params_per_gaussian
        self.window = accumulation_window
        self.damping = damping
        self.device = device
        
        # Accumulated diagonal of the Gauss-Newton Hessian
        # Shape: [num_gaussians, num_params] — grows with map
        self._diag_hessian_sum: Optional[torch.Tensor] = None
        self._frame_count: int = 0
    
    def accumulate(
        self,
        loss: torch.Tensor,
        gaussian_params: torch.Tensor,  # shape: [N, num_params]
    ) -> None:
        """
        Accumulate Hessian diagonal from one rendered frame.
        
        Called AFTER the rendering loss backward pass. We compute:
            diag(H_k) += (dL/dθ_k)² 
        
        This is the Gauss-Newton approximation: H ≈ J^T J,
        and its diagonal is the squared gradient.
        """
        if gaussian_params.grad is None:
            raise ValueError(
                "gaussian_params.grad is None. Call loss.backward() first, "
                "with retain_graph=True if needed."
            )
        
        # Squared gradient = diagonal of GGN Hessian
        # This is the key mathematical insight that makes it efficient
        grad_squared = gaussian_params.grad.detach() ** 2  # [N, num_params]
        
        if self._diag_hessian_sum is None:
            self._diag_hessian_sum = grad_squared.clone()
        else:
            # Handle map growth: if new Gaussians were added, pad the accumulator
            if grad_squared.shape[0] > self._diag_hessian_sum.shape[0]:
                padding = torch.zeros(
                    grad_squared.shape[0] - self._diag_hessian_sum.shape[0],
                    self.num_params,
                    device=self.device
                )
                self._diag_hessian_sum = torch.cat([self._diag_hessian_sum, padding], dim=0)
            
            self._diag_hessian_sum[:grad_squared.shape[0]] += grad_squared
        
        self._frame_count += 1
    
    def compute(self) -> torch.Tensor:
        """
        Compute epistemic uncertainty σ_k² for each Gaussian.
        
        Returns:
            uncertainties: shape [N] — the scalar uncertainty per Gaussian
            
        Mathematical derivation:
            H_k = (1/T) Σ_t (dL_t/dθ_k)²           (averaged diagonal GGN)
            σ_k² = Tr(H_k⁻¹)                         (Eq. 5)
                 = Σ_j  1 / H_k[j,j]                 (trace of inverse diagonal)
                 = Σ_j  1 / ((1/T) Σ_t (dL_t/dθ_k[j])²)
        """
        if self._diag_hessian_sum is None or self._frame_count == 0:
            raise RuntimeError("No frames accumulated. Call accumulate() first.")
        
        # Average over frames
        diag_hessian_avg = self._diag_hessian_sum / self._frame_count  # [N, num_params]
        
        # Add damping for numerical stability (prevents 1/0)
        diag_hessian_avg = diag_hessian_avg + self.damping
        
        # σ_k² = Tr(H_k⁻¹) = sum of 1/diagonal entries
        # shape: [N, num_params] → [N]
        uncertainties = (1.0 / diag_hessian_avg).sum(dim=-1)
        
        return uncertainties
    
    def compute_per_component(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute uncertainty broken down by parameter group.
        Useful for heterogeneous sensor analysis (Sub-contribution 1.4).
        
        Returns:
            pos_uncertainty:  [N] — uncertainty in position (μ)
            cov_uncertainty:  [N] — uncertainty in covariance (Σ) 
            color_uncertainty: [N] — uncertainty in color (c)
            opacity_uncertainty: [N] — uncertainty in opacity (α)
        """
        if self._diag_hessian_sum is None:
            raise RuntimeError("No frames accumulated.")
        
        diag = self._diag_hessian_sum / self._frame_count + self.damping
        inv_diag = 1.0 / diag  # [N, num_params]
        
        # Split by parameter groups
        # Indices depend on MonoGS's parameter ordering
        pos_unc   = inv_diag[:, 0:3].sum(dim=-1)    # position: first 3 params
        cov_unc   = inv_diag[:, 3:9].sum(dim=-1)    # covariance: next 6 params  
        color_unc = inv_diag[:, 9:12].sum(dim=-1)   # color: next 3 params
        opa_unc   = inv_diag[:, 12:13].sum(dim=-1)  # opacity: next 1 param
        
        return pos_unc, cov_unc, color_unc, opa_unc
    
    def reset(self) -> None:
        """Reset accumulator for next window. Call after compute()."""
        self._diag_hessian_sum = None
        self._frame_count = 0
    
    def should_compute(self) -> bool:
        """Whether we've accumulated enough frames."""
        return self._frame_count >= self.window


class UncertaintyCalibrationMetrics:
    """
    Measures how well-calibrated our uncertainty estimates are.
    
    A well-calibrated uncertainty means: if we predict σ_k² = x, then
    the actual reconstruction error for Gaussian k should be ~x.
    
    We measure this with Expected Calibration Error (ECE).
    Used in experiments on Aria-Multiagent (real sensor data).
    """
    
    @staticmethod
    def compute_ece(
        predicted_uncertainty: torch.Tensor,  # [N] — our σ_k²
        actual_error: torch.Tensor,           # [N] — measured reconstruction error
        num_bins: int = 10,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Expected Calibration Error.
        
        Partition Gaussians into bins by predicted uncertainty.
        In each bin, compare mean predicted uncertainty vs. mean actual error.
        ECE = weighted average of |predicted - actual| across bins.
        
        Returns:
            ece: scalar ECE value (lower is better, 0 = perfectly calibrated)
            bin_predicted: [num_bins] mean predicted uncertainty per bin
            bin_actual: [num_bins] mean actual error per bin
        """
        # Sort by predicted uncertainty
        sorted_indices = predicted_uncertainty.argsort()
        bin_size = len(sorted_indices) // num_bins
        
        bin_predicted = torch.zeros(num_bins)
        bin_actual = torch.zeros(num_bins)
        bin_counts = torch.zeros(num_bins)
        
        for b in range(num_bins):
            start = b * bin_size
            end = start + bin_size if b < num_bins - 1 else len(sorted_indices)
            idx = sorted_indices[start:end]
            
            bin_predicted[b] = predicted_uncertainty[idx].mean()
            bin_actual[b] = actual_error[idx].mean()
            bin_counts[b] = len(idx)
        
        # Weighted ECE
        weights = bin_counts / bin_counts.sum()
        ece = (weights * (bin_predicted - bin_actual).abs()).sum().item()
        
        return ece, bin_predicted, bin_actual
