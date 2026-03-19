"""
Test Lie algebra utilities (STEP 3 in build order).
Run with: python -m pytest tests/test_lie_algebra.py -v

Required: ALL 7 tests must pass before proceeding to STEP 4.
"""
import torch
import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.consensus.lie_algebra import (
    skew_symmetric,
    so3_exp,
    so3_log,
    se3_exp,
    se3_log,
    se3_adjoint,
)


def random_so3(dtype=torch.float64):
    """Generate a random SO(3) rotation matrix via QR decomposition."""
    A = torch.randn(3, 3, dtype=dtype)
    Q, R_ = torch.linalg.qr(A)
    # Ensure det(Q) = +1
    Q = Q * torch.sign(torch.det(Q))
    return Q


def random_se3(dtype=torch.float64):
    """Generate a random SE(3) matrix."""
    T = torch.eye(4, dtype=dtype)
    T[:3, :3] = random_so3(dtype)
    T[:3, 3] = torch.randn(3, dtype=dtype)
    return T


class TestSkewSymmetric:
    def test_skew_antisymmetric(self):
        """skew(v) + skew(v).T == 0"""
        for _ in range(100):
            v = torch.randn(3, dtype=torch.float64)
            S = skew_symmetric(v)
            assert torch.allclose(S + S.T, torch.zeros(3, 3, dtype=torch.float64), atol=1e-9), \
                f"Skew matrix not antisymmetric for v={v}"


    def test_skew_cross(self):
        """skew(v) @ u == cross(v, u)"""
        for _ in range(100):
            v = torch.randn(3, dtype=torch.float64)
            u = torch.randn(3, dtype=torch.float64)
            assert torch.allclose(skew_symmetric(v) @ u, torch.cross(v, u), atol=1e-9)


class TestSO3:
    def test_so3_exp_at_zero(self):
        """so3_exp(zeros(3)) == I_3"""
        omega = torch.zeros(3, dtype=torch.float64)
        R = so3_exp(omega)
        I = torch.eye(3, dtype=torch.float64)
        assert torch.allclose(R, I, atol=1e-9), \
            f"so3_exp(0) != I:\n{R}"

    def test_so3_is_rotation(self):
        """so3_exp always produces valid rotation (det=1, R^T R=I)."""
        for _ in range(100):
            omega = torch.randn(3, dtype=torch.float64)
            R = so3_exp(omega)
            assert abs(torch.det(R).item() - 1.0) < 1e-5, f"det(R) = {torch.det(R)}"
            assert torch.allclose(R.T @ R, torch.eye(3, dtype=torch.float64), atol=1e-5)

    def test_so3_near_zero(self):
        """No NaN for angle near zero."""
        omega = torch.zeros(3, dtype=torch.float64)
        omega[0] = 1e-8
        assert not torch.isnan(so3_log(so3_exp(omega))).any()

    def test_so3_log_exp_roundtrip(self):
        """so3_exp(so3_log(R)) == R for 1000 random SO(3)"""
        for i in range(1000):
            R = random_so3()
            omega = so3_log(R)
            R_recovered = so3_exp(omega)
            assert torch.allclose(R, R_recovered, atol=1e-6), \
                f"SO(3) roundtrip failed at iteration {i}:\n" \
                f"R = {R}\nomega = {omega}\nR_rec = {R_recovered}\n" \
                f"diff = {(R - R_recovered).abs().max()}"


class TestSE3:
    def test_se3_log_exp_roundtrip(self):
        """se3_exp(se3_log(T)) == T for 1000 random SE(3) transforms"""
        for i in range(1000):
            T = random_se3()
            xi = se3_log(T)
            T_recovered = se3_exp(xi)
            assert torch.allclose(T, T_recovered, atol=1e-6), \
                f"SE(3) roundtrip failed at iteration {i}:\n" \
                f"T = {T}\nxi = {xi}\nT_rec = {T_recovered}\n" \
                f"diff = {(T - T_recovered).abs().max()}"

    def test_se3_log_near_identity(self):
        """No NaN for angle < 1e-7. Result norm should be < 1e-5."""
        for _ in range(100):
            # Small perturbation around identity
            xi_small = torch.randn(6, dtype=torch.float64) * 1e-9
            T_near_id = se3_exp(xi_small)
            xi_back = se3_log(T_near_id)
            assert not torch.isnan(xi_back).any(), \
                f"NaN in se3_log near identity: xi_small={xi_small}, result={xi_back}"
            assert xi_back.norm() < 1e-5, \
                f"se3_log near identity too large: {xi_back.norm()}"

    def test_se3_log_near_pi(self):
        """No crash for angle ≈ π."""
        # Rotation by pi around x-axis
        R_pi_x = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=torch.float64)
        T = torch.eye(4, dtype=torch.float64)
        T[:3, :3] = R_pi_x
        T[:3, 3] = torch.randn(3, dtype=torch.float64)

        # Should not raise an exception
        xi = se3_log(T)
        assert not torch.isnan(xi).any(), f"NaN in se3_log near pi: {xi}"

        # Rotation by pi around y-axis
        R_pi_y = torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=torch.float64)
        T[:3, :3] = R_pi_y
        xi = se3_log(T)
        assert not torch.isnan(xi).any(), f"NaN in se3_log near pi (y-axis): {xi}"

        # Rotation by pi around z-axis
        R_pi_z = torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)
        T[:3, :3] = R_pi_z
        xi = se3_log(T)
        assert not torch.isnan(xi).any(), f"NaN in se3_log near pi (z-axis): {xi}"


class TestAdjoint:
    def test_adjoint_shape(self):
        """se3_adjoint(T).shape == (6, 6)"""
        for _ in range(10):
            T = random_se3()
            Ad = se3_adjoint(T)
            assert Ad.shape == (6, 6), f"Expected shape (6, 6), got {Ad.shape}"
