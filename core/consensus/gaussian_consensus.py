# core/consensus/gaussian_consensus.py
#
# Called AFTER RiemannianADMM has converged and we have the relative transform T_{i<-j}.
# Fuses neighbor's Gaussian map into this robot's map.

import torch
from torch import Tensor
from typing import Tuple, Optional, Dict
from core.types import GaussianMap
from core.uncertainty.propagation import uncertainty_weighted_average, propagate_uncertainty_through_transform
from core.consensus.matching import match_gaussians


def _quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply quaternions in (w, x, y, z) convention. Returns [N, 4]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert [3,3] rotation matrix to (w,x,y,z) quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = torch.stack([w, x, y, z])
    return q / q.norm()


class GaussianConsensus:
    """
    Fuses Gaussian maps from multiple robots after pose consensus converges.
    """

    def __init__(
        self,
        robot_id: int,
        match_distance_thresh: float = 0.05,
        opacity_thresh: float = 0.01,
        max_gaussians: int = 100_000,
        device: str = "cuda",
    ):
        self.robot_id = robot_id
        self.match_distance_thresh = match_distance_thresh
        self.opacity_thresh = opacity_thresh
        self.max_gaussians = max_gaussians
        self.device = device

    def transform_gaussians(
        self,
        gmap: GaussianMap,
        T: Tensor,
        T_cov: Optional[Tensor] = None,
    ) -> GaussianMap:
        """
        Rigidly transform all Gaussians in gmap by T (SE(3): gmap's frame -> this robot's frame).
        """
        R = T[:3, :3]
        t = T[:3, 3]

        # Transform means
        new_means = (R @ gmap.means.T).T + t

        # Transform quaternions: q_new = R_quat * q_old
        R_quat = _rotation_matrix_to_quaternion(R).to(gmap.quats.device)
        R_quat_expanded = R_quat.unsqueeze(0).expand(gmap.quats.shape[0], -1)
        new_quats = _quat_multiply(R_quat_expanded, gmap.quats)
        new_quats = new_quats / new_quats.norm(dim=-1, keepdim=True).clamp(min=1e-10)

        # Scales, opacities, SH DC are invariant under rigid transform
        new_scales = gmap.scales.clone()
        new_opacities = gmap.opacities.clone()
        new_sh_dc = gmap.sh_dc.clone()
        new_sh_rest = gmap.sh_rest.clone()

        # FIM propagation
        new_fim_means = gmap.fim_means
        if gmap.fim_means is not None and T_cov is not None:
            new_fim_means = propagate_uncertainty_through_transform(
                gmap.fim_means, gmap.means, T, T_cov,
            )
        elif gmap.fim_means is not None:
            # No pose uncertainty — just rotate the FIM
            R_sq = R.pow(2)
            var_old = 1.0 / gmap.fim_means.clamp(min=1e-10)
            var_new = var_old @ R_sq.T
            new_fim_means = 1.0 / var_new.clamp(min=1e-10)

        return GaussianMap(
            means=new_means,
            quats=new_quats,
            scales=new_scales,
            opacities=new_opacities,
            sh_dc=new_sh_dc,
            sh_rest=new_sh_rest,
            robot_id=gmap.robot_id,
            timestamp=gmap.timestamp,
            fim_means=new_fim_means,
            fim_quats=gmap.fim_quats.clone() if gmap.fim_quats is not None else None,
            fim_scales=gmap.fim_scales.clone() if gmap.fim_scales is not None else None,
            fim_opac=gmap.fim_opac.clone() if gmap.fim_opac is not None else None,
            fim_sh_dc=gmap.fim_sh_dc.clone() if gmap.fim_sh_dc is not None else None,
        )

    def fuse(
        self,
        local_map: GaussianMap,
        neighbor_map: GaussianMap,
    ) -> GaussianMap:
        """
        Fuse neighbor_map into local_map. neighbor_map must already be in local frame.
        """
        matched_local, matched_neighbor, unmatched_local, unmatched_neighbor = match_gaussians(
            local_map.means, neighbor_map.means,
            local_map.opacities, neighbor_map.opacities,
            self.match_distance_thresh, self.opacity_thresh,
        )

        field_names = ['means', 'quats', 'scales', 'opacities', 'sh_dc']
        fim_names = ['fim_means', 'fim_quats', 'fim_scales', 'fim_opac', 'fim_sh_dc']
        fim_key_map = {
            'means': 'fim_means', 'quats': 'fim_quats', 'scales': 'fim_scales',
            'opacities': 'fim_opac', 'sh_dc': 'fim_sh_dc',
        }

        parts_list = {name: [] for name in field_names}
        fim_parts_list = {name: [] for name in fim_names}

        if len(matched_local) > 0:
            # Extract matched params
            params_a = {name: getattr(local_map, name)[matched_local] for name in field_names}
            params_b = {name: getattr(neighbor_map, name)[matched_neighbor] for name in field_names}

            def _get_fim(gmap, indices, field, param_name):
                f = getattr(gmap, field)
                if f is not None:
                    return f[indices]
                return torch.ones_like(getattr(gmap, param_name)[indices])

            fim_a = {name: _get_fim(local_map, matched_local, fim_key_map[name], name) for name in field_names}
            fim_b = {name: _get_fim(neighbor_map, matched_neighbor, fim_key_map[name], name) for name in field_names}

            fused_params, fused_fim = uncertainty_weighted_average(params_a, params_b, fim_a, fim_b)

            for name in field_names:
                parts_list[name].append(fused_params[name])
            for name in field_names:
                fim_parts_list[fim_key_map[name]].append(fused_fim[name])

        # Unmatched local
        if len(unmatched_local) > 0:
            for name in field_names:
                parts_list[name].append(getattr(local_map, name)[unmatched_local])
            for name in field_names:
                f = getattr(local_map, fim_key_map[name])
                if f is not None:
                    fim_parts_list[fim_key_map[name]].append(f[unmatched_local])
                else:
                    fim_parts_list[fim_key_map[name]].append(
                        torch.ones_like(getattr(local_map, name)[unmatched_local])
                    )

        # Unmatched neighbor
        if len(unmatched_neighbor) > 0:
            for name in field_names:
                parts_list[name].append(getattr(neighbor_map, name)[unmatched_neighbor])
            for name in field_names:
                f = getattr(neighbor_map, fim_key_map[name])
                if f is not None:
                    fim_parts_list[fim_key_map[name]].append(f[unmatched_neighbor])
                else:
                    fim_parts_list[fim_key_map[name]].append(
                        torch.ones_like(getattr(neighbor_map, name)[unmatched_neighbor])
                    )

        def _cat_or_empty(lst, ref_tensor):
            if lst:
                return torch.cat(lst, dim=0)
            return ref_tensor[:0]

        fused_map = GaussianMap(
            means=_cat_or_empty(parts_list['means'], local_map.means),
            quats=_cat_or_empty(parts_list['quats'], local_map.quats),
            scales=_cat_or_empty(parts_list['scales'], local_map.scales),
            opacities=_cat_or_empty(parts_list['opacities'], local_map.opacities),
            sh_dc=_cat_or_empty(parts_list['sh_dc'], local_map.sh_dc),
            sh_rest=torch.cat([
                local_map.sh_rest[torch.cat([matched_local, unmatched_local])] if len(matched_local) + len(unmatched_local) > 0 else local_map.sh_rest[:0],
                neighbor_map.sh_rest[unmatched_neighbor] if len(unmatched_neighbor) > 0 else neighbor_map.sh_rest[:0],
            ], dim=0) if local_map.sh_rest.shape[0] + neighbor_map.sh_rest.shape[0] > 0 else local_map.sh_rest[:0],
            robot_id=local_map.robot_id,
            timestamp=max(local_map.timestamp, neighbor_map.timestamp),
            fim_means=_cat_or_empty(fim_parts_list['fim_means'], local_map.means),
            fim_quats=_cat_or_empty(fim_parts_list['fim_quats'], local_map.quats),
            fim_scales=_cat_or_empty(fim_parts_list['fim_scales'], local_map.scales),
            fim_opac=_cat_or_empty(fim_parts_list['fim_opac'], local_map.opacities),
            fim_sh_dc=_cat_or_empty(fim_parts_list['fim_sh_dc'], local_map.sh_dc),
        )

        return self.prune(fused_map)

    def prune(self, gmap: GaussianMap) -> GaussianMap:
        """
        Remove low-quality Gaussians:
          - sigmoid(opacity) < 0.01
          - FIM trace < 1e-6 (completely unconstrained)
          - exp(scale).max > 2.0 meters (spuriously large)

        If N > max_gaussians, keep top-N by FIM trace.
        """
        N = gmap.means.shape[0]
        if N == 0:
            return gmap

        keep = torch.ones(N, dtype=torch.bool, device=gmap.means.device)

        # Opacity filter
        opacity_prob = torch.sigmoid(gmap.opacities).squeeze(-1)
        keep = keep & (opacity_prob >= 0.01)

        # FIM trace filter
        if gmap.fim_means is not None:
            fim_trace = gmap.fim_means.sum(dim=-1)
            if gmap.fim_quats is not None:
                fim_trace = fim_trace + gmap.fim_quats.sum(dim=-1)
            if gmap.fim_scales is not None:
                fim_trace = fim_trace + gmap.fim_scales.sum(dim=-1)
            if gmap.fim_opac is not None:
                fim_trace = fim_trace + gmap.fim_opac.squeeze(-1)
            if gmap.fim_sh_dc is not None:
                fim_trace = fim_trace + gmap.fim_sh_dc.sum(dim=(-2, -1))
            keep = keep & (fim_trace >= 1e-6)

        # Scale filter
        max_scale = torch.exp(gmap.scales).max(dim=-1).values
        keep = keep & (max_scale <= 2.0)

        # Apply mask
        idx = torch.where(keep)[0]

        # If still too many, keep top by FIM trace
        if len(idx) > self.max_gaussians and gmap.fim_means is not None:
            traces = fim_trace[idx]
            _, top_indices = traces.topk(self.max_gaussians)
            idx = idx[top_indices]

        return self._index_map(gmap, idx)

    def _index_map(self, gmap: GaussianMap, idx: Tensor) -> GaussianMap:
        """Select a subset of Gaussians by index."""
        def _sel(t, i):
            return t[i] if t is not None else None

        return GaussianMap(
            means=gmap.means[idx],
            quats=gmap.quats[idx],
            scales=gmap.scales[idx],
            opacities=gmap.opacities[idx],
            sh_dc=gmap.sh_dc[idx],
            sh_rest=gmap.sh_rest[idx],
            robot_id=gmap.robot_id,
            timestamp=gmap.timestamp,
            fim_means=_sel(gmap.fim_means, idx),
            fim_quats=_sel(gmap.fim_quats, idx),
            fim_scales=_sel(gmap.fim_scales, idx),
            fim_opac=_sel(gmap.fim_opac, idx),
            fim_sh_dc=_sel(gmap.fim_sh_dc, idx),
        )
