# core/uncertainty/propagation.py

import torch
from torch import Tensor
from typing import Dict, Tuple
from core.consensus.lie_algebra import skew_symmetric


def propagate_uncertainty_through_transform(
    fim_means: Tensor,
    means: Tensor,
    transform: Tensor,
    transform_cov: Tensor,
) -> Tensor:
    """
    First-order propagation of Gaussian position uncertainty through a rigid transform.

    Args:
        fim_means:     [N, 3] FIM diagonal for Gaussian positions (source frame)
        means:         [N, 3] Gaussian positions in source frame
        transform:     [4, 4] SE(3): source -> target
        transform_cov: [6, 6] covariance of transform in se(3)

    Returns:
        [N, 3] updated FIM diagonal for positions (in target frame).
        Lower values = higher uncertainty.
    """
    R = transform[:3, :3]  # [3, 3]
    t = transform[:3, 3]   # [3]
    N = means.shape[0]

    # Variance from FIM diagonal
    var_old = 1.0 / fim_means.clamp(min=1e-10)  # [N, 3]

    # Rotate variance: diag(R @ diag(var_i) @ R^T) for each Gaussian
    # = sum_j R_{k,j}^2 * var_i_j  for each output dim k
    R_sq = R.pow(2)  # [3, 3]
    var_rot = var_old @ R_sq.T  # [N, 3]

    # Transform means to target frame
    mu_new = (R @ means.T).T + t  # [N, 3]

    # Pose contribution: J_ti @ transform_cov @ J_ti^T for each Gaussian
    # J_ti = [I_3 | -skew(mu_new[i])]  shape [3, 6]
    # Convention: xi = [omega(3), v(3)] -> rotation first, translation second
    # So J_ti = [-skew(mu_new[i]) | I_3]  (omega part | translation part)
    I3 = torch.eye(3, device=means.device, dtype=means.dtype)

    var_pose = torch.zeros(N, 3, device=means.device, dtype=means.dtype)
    for i in range(N):
        skew_mu = skew_symmetric(mu_new[i])  # [3, 3]
        J_ti = torch.cat([-skew_mu, I3], dim=1)  # [3, 6]
        cov_i = J_ti @ transform_cov @ J_ti.T  # [3, 3]
        var_pose[i] = cov_i.diag()

    var_new = var_rot + var_pose
    return 1.0 / var_new.clamp(min=1e-10)  # [N, 3]


def uncertainty_weighted_average(
    params_a: Dict[str, Tensor],
    params_b: Dict[str, Tensor],
    fim_a: Dict[str, Tensor],
    fim_b: Dict[str, Tensor],
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Fuse matched Gaussian pairs using FIM-weighted averaging.

    For each parameter dimension k:
        theta_fused_k = (lambda_Ak * theta_Ak + lambda_Bk * theta_Bk)
                        / (lambda_Ak + lambda_Bk)

    Special handling:
        - 'means': plain weighted average in R^3
        - 'scales': weighted average (MonoGS stores log-scales)
        - 'opacities': weighted average (MonoGS stores logit-opacities)
        - 'sh_dc': weighted average per channel
        - 'quats': SLERP with weight based on mean FIM

    Args:
        params_a, params_b: matched Gaussian params (same N, same index = same pair)
        fim_a, fim_b: FIM diagonals

    Returns:
        (params_fused, fim_fused) where fim_fused[k] = fim_a[k] + fim_b[k]
    """
    params_fused = {}
    fim_fused = {}

    for key in params_a:
        fa = fim_a[key]
        fb = fim_b[key]
        pa = params_a[key]
        pb = params_b[key]

        total = fa + fb
        fim_fused[key] = total

        if key == 'quats':
            # SLERP: weight = mean FIM of B / (mean FIM of A + mean FIM of B)
            lam_a = fa.mean(dim=-1, keepdim=True)  # [N, 1]
            lam_b = fb.mean(dim=-1, keepdim=True)  # [N, 1]
            w = lam_b / (lam_a + lam_b).clamp(min=1e-10)  # [N, 1]
            fused = _slerp(pa, pb, w)
            params_fused[key] = fused
        else:
            # Weighted average
            safe_total = total.clamp(min=1e-10)
            fused = (fa * pa + fb * pb) / safe_total
            params_fused[key] = fused

    return params_fused, fim_fused


def _slerp(q0: Tensor, q1: Tensor, t: Tensor) -> Tensor:
    """
    Spherical linear interpolation between quaternions.

    Args:
        q0: [N, 4] quaternions (w, x, y, z)
        q1: [N, 4] quaternions (w, x, y, z)
        t:  [N, 1] interpolation weight (0 = q0, 1 = q1)

    Returns:
        [N, 4] interpolated unit quaternions
    """
    # Normalize inputs
    q0 = q0 / q0.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    q1 = q1 / q1.norm(dim=-1, keepdim=True).clamp(min=1e-10)

    # Dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)  # [N, 1]

    # If dot < 0, negate q1 to take shorter path
    neg_mask = dot < 0
    q1 = torch.where(neg_mask, -q1, q1)
    dot = torch.where(neg_mask, -dot, dot)

    # Clamp to avoid numerical issues with acos
    dot = dot.clamp(-1.0, 1.0)

    # For very close quaternions, use linear interpolation
    linear_mask = dot > 0.9995

    # SLERP
    omega = torch.acos(dot)  # [N, 1]
    sin_omega = torch.sin(omega).clamp(min=1e-10)

    s0 = torch.sin((1.0 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega

    # Use lerp for nearly-identical quaternions
    s0 = torch.where(linear_mask, 1.0 - t, s0)
    s1 = torch.where(linear_mask, t, s1)

    result = s0 * q0 + s1 * q1
    return result / result.norm(dim=-1, keepdim=True).clamp(min=1e-10)
