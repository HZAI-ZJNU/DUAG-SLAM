# core/consensus/matching.py

import torch
from torch import Tensor
from typing import Tuple


def match_gaussians(
    means_a:     Tensor,
    means_b:     Tensor,
    opacities_a: Tensor,
    opacities_b: Tensor,
    distance_thresh: float = 0.05,
    opacity_thresh:  float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Match Gaussians from map A to map B by nearest-neighbor in 3D.

    Only Gaussians with sigmoid(opacity) > opacity_thresh are candidates.
    Matching is 1-to-1 (bijective).

    Returns:
        matched_a:   [K] LongTensor indices into means_a
        matched_b:   [K] LongTensor indices into means_b (aligned)
        unmatched_a: [N-K] LongTensor
        unmatched_b: [M-K] LongTensor
    """
    N = means_a.shape[0]
    M = means_b.shape[0]
    device = means_a.device

    # Step 1: opacity filter
    mask_a = torch.sigmoid(opacities_a).squeeze(-1) > opacity_thresh
    mask_b = torch.sigmoid(opacities_b).squeeze(-1) > opacity_thresh

    # Original indices of valid Gaussians
    idx_a = torch.where(mask_a)[0]  # [N']
    idx_b = torch.where(mask_b)[0]  # [M']

    if len(idx_a) == 0 or len(idx_b) == 0:
        all_a = torch.arange(N, device=device)
        all_b = torch.arange(M, device=device)
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            all_a,
            all_b,
        )

    valid_a = means_a[idx_a]  # [N', 3]
    valid_b = means_b[idx_b]  # [M', 3]

    # Step 3: pairwise distances
    D = torch.cdist(valid_a, valid_b)  # [N', M']

    N_prime = len(idx_a)
    M_prime = len(idx_b)

    # Step 4: matching
    if N_prime * M_prime < 10_000_000:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(D.detach().cpu().numpy())
        row_ind = torch.tensor(row_ind, dtype=torch.long, device=device)
        col_ind = torch.tensor(col_ind, dtype=torch.long, device=device)
    else:
        # Greedy NN
        col_ind = D.argmin(dim=1)  # [N']
        row_ind = torch.arange(N_prime, device=device)
        # Enforce 1-to-1: keep only first occurrence of each col
        seen = {}
        keep_mask = torch.zeros(N_prime, dtype=torch.bool, device=device)
        col_np = col_ind.cpu().tolist()
        for i, c in enumerate(col_np):
            if c not in seen:
                seen[c] = i
                keep_mask[i] = True
        row_ind = row_ind[keep_mask]
        col_ind = col_ind[keep_mask]

    # Step 5: distance threshold
    dists = D[row_ind, col_ind]
    keep = dists < distance_thresh
    row_ind = row_ind[keep]
    col_ind = col_ind[keep]

    # Step 6: map back to original indices
    matched_a = idx_a[row_ind]
    matched_b = idx_b[col_ind]

    # Unmatched
    matched_a_set = set(matched_a.cpu().tolist())
    matched_b_set = set(matched_b.cpu().tolist())
    unmatched_a = torch.tensor(
        [i for i in range(N) if i not in matched_a_set],
        dtype=torch.long, device=device,
    )
    unmatched_b = torch.tensor(
        [i for i in range(M) if i not in matched_b_set],
        dtype=torch.long, device=device,
    )

    return matched_a, matched_b, unmatched_a, unmatched_b
