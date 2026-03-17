"""
Test DUAG-C core components.
Run with: pytest tests/test_consensus.py -v

These tests verify the NOVEL code in /core works correctly
BEFORE integrating any external repos. This is the first thing
you run after setting up the project.
"""
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.types import GaussianSubMap, GaussianAssociation, ConsensusResult
from core.consensus.gaussian_distance import (
    GaussianDistanceMetric, log_euclidean_distance, upper_tri_to_full
)
from core.consensus.duagc_optimizer import DUAGCOptimizer, DUAGCConfig
from core.uncertainty.hessian_estimator import (
    EpistemicUncertaintyEstimator, UncertaintyCalibrationMetrics
)


class TestGaussianDistance:
    """Test the novel d_G metric (Sub-contribution 1.2)."""
    
    def test_zero_distance_for_identical_gaussians(self):
        """d_G(G, G) should be exactly 0."""
        metric = GaussianDistanceMetric()
        pos = torch.randn(10, 3)
        cov = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
        cov_tri = cov[:, torch.triu_indices(3, 3)[0], torch.triu_indices(3, 3)[1]]
        color = torch.rand(10, 3)
        opacity = torch.rand(10, 1)
        
        dist, components = metric.compute(
            pos, cov_tri, color, opacity,
            pos, cov_tri, color, opacity,
        )
        assert torch.allclose(dist, torch.zeros(10), atol=1e-6)
    
    def test_symmetry(self):
        """d_G(G1, G2) should equal d_G(G2, G1)."""
        metric = GaussianDistanceMetric()
        pos1, pos2 = torch.randn(5, 3), torch.randn(5, 3)
        cov_tri = torch.tensor([[1, 0, 0, 1, 0, 1.0]]).expand(5, -1)
        color1, color2 = torch.rand(5, 3), torch.rand(5, 3)
        opa1, opa2 = torch.rand(5, 1), torch.rand(5, 1)
        
        d12, _ = metric.compute(pos1, cov_tri, color1, opa1, pos2, cov_tri, color2, opa2)
        d21, _ = metric.compute(pos2, cov_tri, color2, opa2, pos1, cov_tri, color1, opa1)
        assert torch.allclose(d12, d21, atol=1e-6)
    
    def test_log_euclidean_identity(self):
        """log(I) = 0, so d_LE(I, I) = 0."""
        I = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        dist = log_euclidean_distance(I, I)
        assert torch.allclose(dist, torch.zeros(3), atol=1e-6)
    
    def test_log_euclidean_positive(self):
        """Distance between different SPD matrices should be positive."""
        cov1 = torch.eye(3).unsqueeze(0) * 1.0
        cov2 = torch.eye(3).unsqueeze(0) * 2.0
        dist = log_euclidean_distance(cov1, cov2)
        assert dist.item() > 0
    
    def test_uncertainty_weighting_effect(self):
        """Higher weight should increase the contribution of a Gaussian pair."""
        metric = GaussianDistanceMetric()
        pos1 = torch.zeros(1, 3)
        pos2 = torch.ones(1, 3)
        cov_tri = torch.tensor([[1, 0, 0, 1, 0, 1.0]])
        color = torch.rand(1, 3)
        opa = torch.rand(1, 1)
        
        low_weight = torch.tensor([0.1])
        high_weight = torch.tensor([10.0])
        
        val_low = metric.compute_weighted(pos1, cov_tri, color, opa, pos2, cov_tri, color, opa, low_weight)
        val_high = metric.compute_weighted(pos1, cov_tri, color, opa, pos2, cov_tri, color, opa, high_weight)
        
        assert val_high > val_low  # Higher weight → higher penalty for disagreement


class TestUncertaintyEstimator:
    """Test the Hessian-based uncertainty (Sub-contribution 1.1)."""
    
    def test_accumulation(self):
        """Uncertainty should be computable after accumulating frames."""
        estimator = EpistemicUncertaintyEstimator(num_params_per_gaussian=14, accumulation_window=3)
        
        for _ in range(3):
            params = torch.randn(100, 14, requires_grad=True)
            loss = (params ** 2).sum()
            loss.backward()
            estimator.accumulate(loss, params)
        
        assert estimator.should_compute()
        uncertainties = estimator.compute()
        assert uncertainties.shape == (100,)
        assert (uncertainties > 0).all()
    
    def test_high_gradient_means_low_uncertainty(self):
        """Gaussians with consistently high gradients → low uncertainty (well-observed)."""
        estimator = EpistemicUncertaintyEstimator(num_params_per_gaussian=3, accumulation_window=5)
        
        for _ in range(5):
            params = torch.randn(2, 3, requires_grad=True)
            # Gaussian 0: high loss (large gradient) → well-constrained → LOW uncertainty
            # Gaussian 1: low loss (small gradient) → poorly constrained → HIGH uncertainty
            loss = 100 * params[0].pow(2).sum() + 0.01 * params[1].pow(2).sum()
            loss.backward()
            estimator.accumulate(loss, params)
        
        uncertainties = estimator.compute()
        assert uncertainties[0] < uncertainties[1]  # Low gradient variance = HIGH uncertainty (inverse!)
    
    def test_reset(self):
        """After reset, should_compute returns False."""
        estimator = EpistemicUncertaintyEstimator(accumulation_window=2)
        params = torch.randn(10, 14, requires_grad=True)
        loss = params.sum()
        loss.backward()
        estimator.accumulate(loss, params)
        estimator.accumulate(loss, params)
        estimator.reset()
        assert not estimator.should_compute()


class TestDUAGCOptimizer:
    """Test the DUAG-C optimizer (Sub-contributions 1.2-1.3)."""
    
    def _make_submap(self, robot_id, n=50, offset=None):
        """Create a synthetic sub-map for testing."""
        positions = torch.randn(n, 3)
        if offset is not None:
            positions = positions + offset
        return GaussianSubMap(
            robot_id=robot_id,
            timestamp=0.0,
            positions=positions,
            covariances=torch.tensor([[1, 0, 0, 1, 0, 1.0]]).expand(n, -1).clone(),
            colors=torch.rand(n, 3),
            opacities=torch.rand(n, 1),
            uncertainties=torch.rand(n, 1) * 0.5 + 0.1,  # σ² ∈ [0.1, 0.6]
            keyframe_poses=torch.eye(4).unsqueeze(0),
        )
    
    def test_identical_submaps_converge_instantly(self):
        """If two robots have identical maps, consensus should converge in 1 iteration."""
        submap = self._make_submap(robot_id=0, n=30)
        submap_copy = GaussianSubMap(
            robot_id=1, timestamp=0.0,
            positions=submap.positions.clone(),
            covariances=submap.covariances.clone(),
            colors=submap.colors.clone(),
            opacities=submap.opacities.clone(),
            uncertainties=submap.uncertainties.clone(),
            keyframe_poses=submap.keyframe_poses.clone(),
        )
        associations = GaussianAssociation(
            robot_i=0, robot_j=1,
            indices_i=torch.arange(30),
            indices_j=torch.arange(30),
            match_confidence=torch.ones(30),
        )
        
        optimizer = DUAGCOptimizer(DUAGCConfig(max_iterations=10), device="cpu")
        result = optimizer.optimize(submap, submap_copy, associations)
        
        assert result.converged
        assert result.primal_residual < 1e-3
    
    def test_uncertainty_weighting_improves_result(self):
        """
        When one sub-map has high uncertainty, consensus should favor the other.
        This is the key scientific claim of Sub-contribution 1.3.
        """
        # Robot 0: accurate map (low uncertainty)
        submap_good = self._make_submap(robot_id=0, n=20)
        submap_good.uncertainties = torch.full((20, 1), 0.01)  # Very confident
        
        # Robot 1: noisy map (high uncertainty) with added noise
        submap_noisy = GaussianSubMap(
            robot_id=1, timestamp=0.0,
            positions=submap_good.positions.clone() + torch.randn(20, 3) * 0.5,  # Added noise
            covariances=submap_good.covariances.clone(),
            colors=submap_good.colors.clone(),
            opacities=submap_good.opacities.clone(),
            uncertainties=torch.full((20, 1), 10.0),  # Very uncertain
            keyframe_poses=submap_good.keyframe_poses.clone(),
        )
        
        associations = GaussianAssociation(
            robot_i=0, robot_j=1,
            indices_i=torch.arange(20),
            indices_j=torch.arange(20),
            match_confidence=torch.ones(20),
        )
        
        optimizer = DUAGCOptimizer(DUAGCConfig(max_iterations=10), device="cpu")
        result = optimizer.optimize(submap_good, submap_noisy, associations)
        
        # After consensus, the merged map's positions should be CLOSER to Robot 0's
        # original positions (the confident ones) than to Robot 1's noisy ones
        dist_to_good = (result.merged_gaussians.positions[:20] - submap_good.positions).norm(dim=-1).mean()
        dist_to_noisy = (result.merged_gaussians.positions[:20] - submap_noisy.positions).norm(dim=-1).mean()
        
        assert dist_to_good < dist_to_noisy, (
            f"Consensus should favor confident robot! "
            f"dist_to_good={dist_to_good:.4f}, dist_to_noisy={dist_to_noisy:.4f}"
        )


class TestCalibration:
    """Test the uncertainty calibration metrics."""
    
    def test_perfect_calibration(self):
        """If predicted uncertainty matches actual error, ECE should be ~0."""
        predicted = torch.linspace(0.1, 1.0, 100)
        actual = predicted + torch.randn(100) * 0.01  # Small noise
        
        ece, _, _ = UncertaintyCalibrationMetrics.compute_ece(predicted, actual, num_bins=5)
        assert ece < 0.1  # Should be close to 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
