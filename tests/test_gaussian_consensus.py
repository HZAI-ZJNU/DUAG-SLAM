# tests/test_gaussian_consensus.py
#
# STEP 10 required tests:
#   test_fuse_with_self
#   test_fuse_count_bound
#   test_transform_then_inverse
#   test_prune_removes_low_opacity

import torch
import pytest
from core.types import GaussianMap
from core.consensus.gaussian_consensus import GaussianConsensus


def _make_test_map(n: int, robot_id: int = 0) -> GaussianMap:
    """Create a synthetic GaussianMap for testing."""
    return GaussianMap(
        means=torch.randn(n, 3),
        quats=torch.nn.functional.normalize(torch.randn(n, 4), dim=-1),
        scales=torch.randn(n, 3) * 0.1 - 2.0,  # log-scale ~ exp(-2) ≈ 0.13m
        opacities=torch.ones(n, 1) * 2.0,  # sigmoid(2) ≈ 0.88
        sh_dc=torch.randn(n, 1, 3),
        sh_rest=torch.randn(n, 15, 3),
        robot_id=robot_id,
        timestamp=0.0,
        fim_means=torch.rand(n, 3) + 0.1,
        fim_quats=torch.rand(n, 4) + 0.1,
        fim_scales=torch.rand(n, 3) + 0.1,
        fim_opac=torch.rand(n, 1) + 0.1,
        fim_sh_dc=torch.rand(n, 1, 3) + 0.1,
    )


def test_fuse_with_self():
    """fuse(map, map) ≈ map (fusing identical maps should approximate original)."""
    n = 30
    gmap = _make_test_map(n)
    gc = GaussianConsensus(robot_id=0, match_distance_thresh=0.01, device="cpu")

    fused = gc.fuse(gmap, gmap)

    # All N should be matched (identical means), fused means ≈ original means
    # After fusion, K matched + 0 unmatched = N total (or less after pruning)
    assert fused.means.shape[0] <= n, "Fused should not have more than N Gaussians"
    # The fused means should approximate the original (weighted avg of two identical = same)
    # We need to match them up since ordering might differ
    # Use manual pairwise L2 instead of torch.cdist (which is numerically unstable near zero)
    diffs = gmap.means.unsqueeze(1) - fused.means.unsqueeze(0)  # [N, M, 3]
    dists = diffs.pow(2).sum(dim=-1).sqrt()  # [N, M]
    closest_dist = dists.min(dim=1).values
    assert torch.allclose(closest_dist, torch.zeros_like(closest_dist), atol=1e-4), (
        f"Fused means deviate from original. Max dist: {closest_dist.max()}"
    )


def test_fuse_count_bound():
    """Result has <= N_a + N_b Gaussians."""
    n_a, n_b = 20, 30
    map_a = _make_test_map(n_a, robot_id=0)
    map_b = _make_test_map(n_b, robot_id=1)

    gc = GaussianConsensus(robot_id=0, device="cpu")
    fused = gc.fuse(map_a, map_b)

    assert fused.means.shape[0] <= n_a + n_b, (
        f"Fused has {fused.means.shape[0]} > {n_a + n_b} Gaussians"
    )


def test_transform_then_inverse():
    """transform(transform(map, T), T^{-1}) ≈ map."""
    n = 25
    gmap = _make_test_map(n)
    gc = GaussianConsensus(robot_id=0, device="cpu")

    # Create a random SE(3) transform
    from core.consensus.lie_algebra import se3_exp, se3_log
    xi = torch.tensor([0.1, -0.2, 0.3, 0.5, -0.3, 0.1])  # small transform
    T = se3_exp(xi)
    T_inv = torch.inverse(T)

    # Transform forward then back
    transformed = gc.transform_gaussians(gmap, T)
    recovered = gc.transform_gaussians(transformed, T_inv)

    assert torch.allclose(recovered.means, gmap.means, atol=1e-4), (
        f"Means not recovered. Max diff: {(recovered.means - gmap.means).abs().max()}"
    )


def test_prune_removes_low_opacity():
    """Gaussian with sigmoid(opacity)=0.001 is removed."""
    n = 20
    gmap = _make_test_map(n)

    # Set some Gaussians to very low opacity (logit corresponding to ~0.001)
    # sigmoid^{-1}(0.001) = log(0.001/0.999) ≈ -6.9
    gmap.opacities[5] = -7.0
    gmap.opacities[10] = -7.0
    gmap.opacities[15] = -7.0

    gc = GaussianConsensus(robot_id=0, device="cpu")
    pruned = gc.prune(gmap)

    assert pruned.means.shape[0] < n, (
        f"Pruned map has {pruned.means.shape[0]} Gaussians, expected < {n}"
    )
    # Should have exactly 3 fewer
    assert pruned.means.shape[0] == n - 3, (
        f"Expected {n-3} Gaussians after pruning, got {pruned.means.shape[0]}"
    )


def test_transform_identity():
    """Transform by identity leaves map unchanged."""
    n = 10
    gmap = _make_test_map(n)
    gc = GaussianConsensus(robot_id=0, device="cpu")
    transformed = gc.transform_gaussians(gmap, torch.eye(4))
    assert torch.allclose(gmap.means, transformed.means, atol=1e-6)


def test_quat_unit_norm():
    """Quaternions remain unit-norm after transform."""
    from core.consensus.gaussian_consensus import _rotation_matrix_to_quaternion
    from core.consensus.lie_algebra import se3_exp

    T = se3_exp(torch.tensor([0.5, 0.3, -0.2, 0.0, 0.0, 0.0]))
    q = _rotation_matrix_to_quaternion(T[:3, :3])
    assert abs(q.norm().item() - 1.0) < 1e-6
