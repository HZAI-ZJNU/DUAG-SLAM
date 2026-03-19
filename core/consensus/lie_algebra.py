# core/consensus/lie_algebra.py
#
# SE(3) Lie group / Lie algebra utilities.
#
# CONVENTION: Matches DPGO's C++ convention.
# Verified against repos/dpgo/include/DPGO/ header files.
# Standard robotics convention: xi = [omega (3); v (3)]
#   omega = rotation axis-angle  (se(3) rotation component)
#   v     = translational velocity (se(3) translation component)
#
# All functions are numerically stable at identity (theta -> 0) via Taylor expansion.

import torch
from torch import Tensor
import math


def skew_symmetric(v: Tensor) -> Tensor:
    """
    3-vector -> 3x3 skew-symmetric matrix.
    [v]_x @ u = cross(v, u)
    v: [3]  ->  [3, 3]
    """
    assert v.shape == (3,), f"Expected shape (3,), got {v.shape}"
    x, y, z = v[0], v[1], v[2]
    z_ = v.new_zeros(1).squeeze()
    return torch.stack([
        torch.stack([z_, -z, y]),
        torch.stack([z, z_, -x]),
        torch.stack([-y, x, z_]),
    ])


def so3_exp(omega: Tensor) -> Tensor:
    """
    SO(3) matrix exponential (Rodrigues formula).
    omega: [3] axis-angle  ->  [3, 3] rotation matrix

    R = I + (sin(theta)/theta)*K + ((1-cos(theta))/theta^2)*K^2
    where K = skew_symmetric(omega), theta = ||omega||

    Stable at theta < 1e-7 via Taylor:
        sin(theta)/theta        -> 1 - theta^2/6
        (1-cos(theta))/theta^2  -> 0.5 - theta^2/24
    """
    theta = omega.norm()
    K = skew_symmetric(omega)
    I = torch.eye(3, dtype=omega.dtype, device=omega.device)
    if theta < 1e-7:
        return I + K + 0.5 * (K @ K)
    s = torch.sin(theta) / theta
    c = (1.0 - torch.cos(theta)) / (theta ** 2)
    return I + s * K + c * (K @ K)


def so3_log(R: Tensor) -> Tensor:
    """
    SO(3) matrix logarithm.
    R: [3, 3]  ->  [3] axis-angle vector

    theta = arccos( (trace(R) - 1) / 2 )
    omega = theta / (2*sin(theta)) * [R32-R23, R13-R31, R21-R12]

    Special cases:
    - theta = 0: return zeros(3)
    - theta = pi: eigendecomposition of (R+I)/2 to find rotation axis
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos_t = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.acos(cos_t)

    if theta.abs() < 1e-7:
        return torch.zeros(3, dtype=R.dtype, device=R.device)

    if (theta - math.pi).abs() < 1e-4:
        # theta = pi: axis is eigenvector of R with eigenvalue +1
        # (R + I) / 2 = axis * axis^T  (rank-1 SPD)
        # Take the column of (R+I)/2 with largest norm
        M = (R + torch.eye(3, dtype=R.dtype, device=R.device)) / 2.0
        norms = M.pow(2).sum(dim=0)
        col = M[:, norms.argmax()]
        axis = col / col.norm()
        return math.pi * axis

    vec = torch.stack([R[2, 1] - R[1, 2],
                       R[0, 2] - R[2, 0],
                       R[1, 0] - R[0, 1]])
    return (theta / (2.0 * torch.sin(theta))) * vec


def se3_exp(xi: Tensor) -> Tensor:
    """
    SE(3) exponential: se(3) -> SE(3).
    xi: [6] = [omega(3), v(3)]   rotation first, translation second
    Returns: [4, 4] SE(3) matrix

    R = so3_exp(omega)
    V = I + ((1-cos(t))/t^2)*K + ((t-sin(t))/t^3)*K^2
    t = V @ v
    T = [[R, t], [0, 1]]

    Stable at t < 1e-7:
        V -> I + 0.5*K
    """
    omega, v = xi[:3], xi[3:]
    theta = omega.norm()
    R = so3_exp(omega)
    K = skew_symmetric(omega)
    I = torch.eye(3, dtype=xi.dtype, device=xi.device)
    if theta < 1e-7:
        V = I + 0.5 * K
    else:
        V = (I
             + ((1.0 - torch.cos(theta)) / theta**2) * K
             + ((theta - torch.sin(theta)) / theta**3) * (K @ K))
    t = V @ v
    T = torch.eye(4, dtype=xi.dtype, device=xi.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T: Tensor) -> Tensor:
    """
    SE(3) logarithm: SE(3) -> se(3).
    T: [4, 4]  ->  [6] = [omega(3), v(3)]

    omega = so3_log(R)
    V_inv @ t = v    where V is as in se3_exp
    V^{-1} = I - 0.5*K + (1/t^2 - (1+cos(t))/(2*t*sin(t)))*K^2
    """
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log(R)
    theta = omega.norm()
    K = skew_symmetric(omega)
    I = torch.eye(3, dtype=T.dtype, device=T.device)
    if theta < 1e-7:
        V_inv = I - 0.5 * K
    else:
        c = (1.0 / theta**2
             - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta)))
        V_inv = I - 0.5 * K + c * (K @ K)
    v = V_inv @ t
    return torch.cat([omega, v])


def se3_adjoint(T: Tensor) -> Tensor:
    """
    SE(3) adjoint map Ad_T in R^{6x6}.
    Transforms a tangent vector from one reference frame to another.
    Used in Riemannian gradient computation.

    For xi = [omega; v] convention:
        Ad_T = [[ R,       0  ],
                [ [t]_x R, R  ]]

    T: [4, 4]  ->  [6, 6]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    tx = skew_symmetric(t)
    Ad = torch.zeros(6, 6, dtype=T.dtype, device=T.device)
    Ad[:3, :3] = R
    Ad[3:, :3] = tx @ R
    Ad[3:, 3:] = R
    return Ad
