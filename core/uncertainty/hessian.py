# core/uncertainty/hessian.py
#
# Per-Gaussian diagonal Fisher Information Matrix via the Gauss-Newton
# (outer-product) approximation: Lambda_k ≈ sum_b (dL/dtheta_k)^2

import torch
from torch import Tensor
from typing import Dict

from extracted.gs_slam.renderer import render
from extracted.gs_slam.loss_utils import l1_loss


def compute_gaussian_fim(
    gaussian_model,
    keyframe_rgbs: Tensor,
    keyframe_cameras: list,
    rasterizer_pipe,
    background: Tensor,
    max_gaussians: int = 50000,
) -> Dict[str, Tensor]:
    """
    Compute per-Gaussian diagonal FIM by accumulating squared gradients
    of the photometric loss over all keyframes.

    Args:
        gaussian_model: extracted.gs_slam.gaussian_model.GaussianModel instance
        keyframe_rgbs:  [B, H, W, 3] float32 in [0,1]
        keyframe_cameras: list of B MonoGS Camera objects
        rasterizer_pipe: MonoGS PipelineParams config object
        background: [3] background color tensor
        max_gaussians: subsample if N > this

    Returns:
        dict mapping parameter name to FIM diagonal tensor (same shape as param)
        All entries >= 0. Zero means the Gaussian was not visible in any keyframe.
    """
    params = gaussian_model.get_fim_params()
    fim = {name: torch.zeros_like(p) for name, p in params.items()}

    B = keyframe_rgbs.shape[0]

    for b in range(B):
        cam_b = keyframe_cameras[b]
        rgb_b = keyframe_rgbs[b]  # [H, W, 3]

        # Ensure params have grad tracking enabled
        for p in params.values():
            p.requires_grad_(True)
            if p.grad is not None:
                p.grad.zero_()

        render_pkg = render(cam_b, gaussian_model, rasterizer_pipe, background)
        rendered = render_pkg["render"]  # [3, H, W]

        # l1_loss expects (network_output, gt) — both in [3, H, W]
        rgb_b_chw = rgb_b.permute(2, 0, 1)  # [H,W,3] -> [3,H,W]
        loss = l1_loss(rendered, rgb_b_chw)

        grads = torch.autograd.grad(
            loss,
            list(params.values()),
            retain_graph=False,
            allow_unused=True,
            create_graph=False,
        )

        for name, g in zip(params.keys(), grads):
            if g is not None:
                fim[name] = fim[name] + g.detach().pow(2)

    return fim
