# tests/test_matching.py
#
# STEP 9 required tests:
#   test_match_identical_maps
#   test_match_distance_threshold
#   test_match_opacity_filter
#   test_match_1to1

import torch
import pytest
from core.consensus.matching import match_gaussians


def test_match_identical_maps():
    """Matching map with itself: K=N, all matched."""
    N = 30
    means = torch.randn(N, 3)
    opacities = torch.ones(N, 1) * 2.0  # sigmoid(2) ≈ 0.88 > 0.1

    matched_a, matched_b, unmatched_a, unmatched_b = match_gaussians(
        means, means, opacities, opacities,
        distance_thresh=0.01,
    )

    assert len(matched_a) == N, f"Expected K={N}, got K={len(matched_a)}"
    assert len(unmatched_a) == 0
    assert len(unmatched_b) == 0


def test_match_distance_threshold():
    """No match if distance > thresh."""
    N = 20
    means_a = torch.zeros(N, 3)
    means_b = torch.ones(N, 3) * 100.0  # Very far apart
    opacities = torch.ones(N, 1) * 2.0

    matched_a, matched_b, unmatched_a, unmatched_b = match_gaussians(
        means_a, means_b, opacities, opacities,
        distance_thresh=0.05,
    )

    assert len(matched_a) == 0, f"Expected K=0, got K={len(matched_a)}"
    assert len(unmatched_a) == N
    assert len(unmatched_b) == N


def test_match_opacity_filter():
    """Low-opacity Gaussians not matched."""
    N = 20
    means = torch.randn(N, 3)

    # Half high opacity, half very low
    opacities = torch.ones(N, 1) * 2.0
    opacities[10:] = -10.0  # sigmoid(-10) ≈ 4.5e-5 < 0.1

    matched_a, matched_b, _, _ = match_gaussians(
        means, means, opacities, opacities,
        distance_thresh=0.01,
    )

    # Only the first 10 (high opacity) should match
    assert len(matched_a) == 10, f"Expected 10 matches, got {len(matched_a)}"

    # Check all matched indices are in the high-opacity range [0..9]
    assert (matched_a < 10).all(), "Matched indices should be in high-opacity range"
    assert (matched_b < 10).all(), "Matched indices should be in high-opacity range"


def test_match_1to1():
    """No Gaussian appears in matched_a or matched_b twice (bijective matching)."""
    N = 50
    means_a = torch.randn(N, 3)
    means_b = means_a + torch.randn(N, 3) * 0.001  # Slight perturbation
    opacities = torch.ones(N, 1) * 2.0

    matched_a, matched_b, _, _ = match_gaussians(
        means_a, means_b, opacities, opacities,
        distance_thresh=0.1,
    )

    # Check uniqueness
    assert len(matched_a.unique()) == len(matched_a), (
        f"matched_a has duplicates: {len(matched_a)} entries, {len(matched_a.unique())} unique"
    )
    assert len(matched_b.unique()) == len(matched_b), (
        f"matched_b has duplicates: {len(matched_b)} entries, {len(matched_b.unique())} unique"
    )


def test_match_covers_all():
    """matched + unmatched = full index set for both maps."""
    N, M = 15, 10
    means_a = torch.randn(N, 3)
    means_b = torch.randn(M, 3)
    opacities_a = torch.ones(N, 1) * 2.0
    opacities_b = torch.ones(M, 1) * 2.0

    matched_a, matched_b, unmatched_a, unmatched_b = match_gaussians(
        means_a, means_b, opacities_a, opacities_b,
    )

    assert sorted(matched_a.tolist() + unmatched_a.tolist()) == list(range(N))
    assert sorted(matched_b.tolist() + unmatched_b.tolist()) == list(range(M))
