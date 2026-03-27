# core/pipeline/local_slam_wrapper.py
#
# Wraps MonoGS's tracking+mapping loop to provide the interface expected by
# RobotNode.  Runs in single-process / single-thread mode (no multiprocessing)
# for simulation experiments, calling tracking and mapping directly each frame.
#
# Public interface consumed by RobotNode:
#   local_slam.process_frame(rgb, depth, timestamp)
#   local_slam.get_current_pose()       -> torch.Tensor [4,4] world-to-camera
#   local_slam.get_gaussian_map()       -> GaussianMap
#   local_slam.get_current_camera()     -> Camera
#   local_slam.gaussian_model           -> GaussianModel
#   local_slam.rasterizer_pipe          -> munchified pipeline_params
#   local_slam.background               -> torch.Tensor [3]

import torch
import numpy as np
from typing import Optional
from munch import munchify

from core.types import GaussianMap
from extracted.gs_slam.gaussian_model import GaussianModel
from extracted.gs_slam.camera_utils import Camera
from extracted.gs_slam.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
from extracted.gs_slam.renderer import render
from extracted.gs_slam.pose_utils import update_pose
from extracted.gs_slam.slam_utils import (
    get_loss_tracking,
    get_loss_mapping,
    get_median_depth,
)


class LocalSLAMWrapper:
    """
    Single-robot MonoGS wrapper.  One instance per RobotNode.

    Usage::

        wrapper = LocalSLAMWrapper(config, device="cuda")
        # then each frame:
        wrapper.process_frame(rgb, depth, timestamp)
        pose  = wrapper.get_current_pose()   # [4,4] world-to-camera
        gmap  = wrapper.get_gaussian_map()   # GaussianMap dataclass
    """

    def __init__(self, config: dict, device: str = "cuda"):
        """
        Args:
            config: Full MonoGS-style YAML config dict (already loaded).
                    Must contain Dataset, Training, opt_params, model_params,
                    pipeline_params sections.
            device: CUDA device string.
        """
        self.device = device
        self.config = config

        # Calibration
        cal = config["Dataset"]["Calibration"]
        self.fx = cal["fx"]
        self.fy = cal["fy"]
        self.cx = cal["cx"]
        self.cy = cal["cy"]
        self.W  = int(cal["width"])
        self.H  = int(cal["height"])
        self.depth_scale = cal.get("depth_scale", 1.0)
        self.fovx = focal2fov(self.fx, self.W)
        self.fovy = focal2fov(self.fy, self.H)

        # Projection matrix (computed once, reused for every Camera)
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            W=self.W, H=self.H,
        ).transpose(0, 1).to(device)

        # Pipeline params (munchified dict like MonoGS expects)
        self.rasterizer_pipe = munchify(config["pipeline_params"])

        # Opt params
        self.opt_params = munchify(config["opt_params"])

        # Background (black for indoor)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

        # Training hyper-params
        train = config["Training"]
        self.tracking_itr_num = train.get("tracking_itr_num", 100)
        self.mapping_itr_num  = train.get("mapping_itr_num", 150)
        self.init_itr_num     = train.get("init_itr_num", 1050)
        self.init_gaussian_update = train.get("init_gaussian_update", 100)
        self.init_gaussian_reset  = train.get("init_gaussian_reset", 500)
        self.init_gaussian_th     = train.get("init_gaussian_th", 0.005)
        self.kf_interval      = train.get("kf_interval", 4)
        self.kf_translation   = train.get("kf_translation", 0.04)
        self.kf_min_translation = train.get("kf_min_translation", 0.02)
        self.kf_overlap       = train.get("kf_overlap", 0.95)
        self.window_size      = train.get("window_size", 10)
        self.gaussian_update_every  = train.get("gaussian_update_every", 150)
        self.gaussian_update_offset = train.get("gaussian_update_offset", 50)
        self.gaussian_th     = train.get("gaussian_th", 0.7)
        self.gaussian_reset  = train.get("gaussian_reset", 2001)
        self.size_threshold  = train.get("size_threshold", 20)
        self.max_gaussians   = train.get("max_gaussians", 80000)

        # Gaussian model
        sh_degree = config["model_params"].get("sh_degree", 0)
        self.gaussian_model = GaussianModel(sh_degree, config=config)
        self.gaussian_model.init_lr(6.0)          # cameras_extent
        self.gaussian_model.training_setup(self.opt_params)

        self.cameras_extent = 6.0
        self.init_gaussian_extent = (
            self.cameras_extent * train.get("init_gaussian_extent", 30)
        )
        self.gaussian_extent = (
            self.cameras_extent * train.get("gaussian_extent", 1.0)
        )

        # State
        self._cameras: dict = {}           # uid -> Camera
        self._current_camera: Optional[Camera] = None
        self._current_pose: Optional[torch.Tensor] = None  # [4,4] w2c
        self._initialized = False
        self._frame_count = 0
        self._iteration_count = 0
        self._kf_indices: list = []
        self._current_window: list = []
        self._occ_aware_visibility: dict = {}
        self._median_depth = 1.0
        self._last_kf_idx = 0

        # Gradient-based FIM accumulator (diagonal Hessian approximation)
        # Accumulated across mapping steps: Λ_k ≈ Σ_b (∂L/∂θ_k)²
        self._fim_means:  Optional[torch.Tensor] = None  # [N, 3]
        self._fim_quats:  Optional[torch.Tensor] = None  # [N, 4]
        self._fim_scales: Optional[torch.Tensor] = None  # [N, 3]
        self._fim_opac:   Optional[torch.Tensor] = None  # [N, 1]

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(
        self, rgb: torch.Tensor, depth: torch.Tensor, timestamp: float
    ) -> None:
        """
        Run one tracking + (optionally) mapping step.

        Args:
            rgb:   [H, W, 3] float32 in [0,1]  OR  [3, H, W] float32 in [0,1]
            depth: [H, W] float32 in meters    OR  [H, W, 1] float32
            timestamp: frame timestamp (seconds)
        """
        # Normalise input shapes
        if rgb.dim() == 3 and rgb.shape[2] == 3:
            rgb = rgb.permute(2, 0, 1)          # -> [3, H, W]
        rgb = rgb.to(self.device)

        if depth.dim() == 3:
            depth = depth.squeeze(-1)           # -> [H, W]
        depth_np = depth.cpu().numpy()          # Camera stores raw numpy depth

        uid = self._frame_count
        viewpoint = self._build_camera(rgb, depth_np, uid)
        self._cameras[uid] = viewpoint

        if not self._initialized:
            self._initialize_map(uid, viewpoint, depth_np)
            self._frame_count += 1
            return

        # ── Tracking ──
        self._tracking(uid, viewpoint)

        # ── Keyframe decision + mapping ──
        vis_filter = self._occ_aware_visibility.get(self._last_kf_idx)
        cur_vis = self._cur_visibility
        is_kf = self._is_keyframe(uid, vis_filter, cur_vis)

        if is_kf:
            depth_map = self._add_new_keyframe(uid, depth_np)
            self._kf_indices.append(uid)
            self._last_kf_idx = uid

            # Add Gaussians from new keyframe
            self.gaussian_model.extend_from_pcd_seq(
                viewpoint, kf_id=uid, init=False, depthmap=depth_map,
            )

            # Update window
            self._current_window = [uid] + self._current_window
            if len(self._current_window) > self.window_size:
                self._current_window = self._current_window[:self.window_size]

            # Run mapping iterations
            self._mapping(self._current_window, iters=self.mapping_itr_num)
        else:
            # Non-keyframe: release GPU tensors (original_image, depth) to prevent
            # O(N_frames) memory growth.  MonoGS does this in slam_frontend.py:311-312.
            viewpoint.clean()

        if self._frame_count % 10 == 0:
            torch.cuda.empty_cache()

        self._frame_count += 1

    def get_current_pose(self) -> torch.Tensor:
        """Returns [4,4] world-to-camera SE(3) matrix."""
        assert self._current_pose is not None
        return self._current_pose.clone()

    def get_current_camera(self):
        """Returns current MonoGS Camera object."""
        return self._current_camera

    def get_gaussian_map(self) -> GaussianMap:
        """Convert GaussianModel state to GaussianMap dataclass."""
        gm = self.gaussian_model
        N = gm._xyz.shape[0]
        if N == 0:
            return GaussianMap(
                means=torch.empty(0, 3, device=self.device),
                quats=torch.empty(0, 4, device=self.device),
                scales=torch.empty(0, 3, device=self.device),
                opacities=torch.empty(0, 1, device=self.device),
                sh_dc=torch.empty(0, 1, 3, device=self.device),
                sh_rest=torch.empty(0, 0, 3, device=self.device),
                robot_id=-1, timestamp=0.0,
            )
        return GaussianMap(
            means     = gm._xyz.detach(),
            quats     = gm._rotation.detach(),
            scales    = gm._scaling.detach(),
            opacities = gm._opacity.detach(),
            sh_dc     = gm._features_dc.detach(),
            sh_rest   = gm._features_rest.detach(),
            robot_id  = -1,
            timestamp = 0.0,
            fim_means  = self._fim_means.detach().clone() if self._fim_means is not None and self._fim_means.shape[0] == N else None,
            fim_quats  = self._fim_quats.detach().clone() if self._fim_quats is not None and self._fim_quats.shape[0] == N else None,
            fim_scales = self._fim_scales.detach().clone() if self._fim_scales is not None and self._fim_scales.shape[0] == N else None,
            fim_opac   = self._fim_opac.detach().clone() if self._fim_opac is not None and self._fim_opac.shape[0] == N else None,
        )

    def as_slam_step(self):
        """
        Return a callable ``(rgb, depth, timestamp) -> (GaussianMap, pose)``
        suitable for injection into RobotNode as ``slam_step``.
        """
        def _step(rgb, depth, timestamp):
            self.process_frame(rgb, depth, timestamp)
            return self.get_gaussian_map(), self.get_current_pose()
        return _step

    # ─────────────────────────────────────────────────────────────────────────
    # Private – Camera construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_camera(self, rgb_chw: torch.Tensor, depth_np: np.ndarray, uid: int) -> Camera:
        """
        Build a MonoGS Camera from an RGB-D frame.

        Args:
            rgb_chw:  [3, H, W] float32 on self.device, values in [0, 1]
            depth_np: [H, W] numpy float32, depth in meters
            uid:      unique frame id
        """
        gt_T = torch.eye(4, device=self.device)
        cam = Camera(
            uid=uid,
            color=rgb_chw,
            depth=depth_np,
            gt_T=gt_T,
            projection_matrix=self.projection_matrix,
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            fovx=self.fovx, fovy=self.fovy,
            image_height=self.H, image_width=self.W,
            device=self.device,
        )
        cam.compute_grad_mask(self.config)
        return cam

    # ─────────────────────────────────────────────────────────────────────────
    # Private – Initialization (first frame)
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_map(self, uid: int, viewpoint: Camera, depth_np: np.ndarray) -> None:
        """Seed the Gaussian map from the first RGB-D frame and run init iterations."""
        # Set camera to identity (ground truth for first frame)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        self._current_camera = viewpoint
        self._current_pose = self._camera_to_w2c(viewpoint)

        # Add first keyframe depth map
        depth_map = self._add_new_keyframe(uid, depth_np)
        self._kf_indices.append(uid)
        self._last_kf_idx = uid
        self._current_window = [uid]

        # Seed Gaussians from depth
        self.gaussian_model.extend_from_pcd_seq(
            viewpoint, kf_id=uid, init=True, depthmap=depth_map,
        )

        # Run initialization mapping iterations
        for it in range(self.init_itr_num):
            self._iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussian_model, self.rasterizer_pipe, self.background,
            )
            if render_pkg is None:
                break

            image = render_pkg["render"]
            depth_r = render_pkg["depth"]
            opacity = render_pkg["opacity"]
            viewspace_pts = render_pkg["viewspace_points"]
            vis_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            n_touched = render_pkg["n_touched"]

            loss = get_loss_mapping(
                self.config, image, depth_r, viewpoint, opacity, initialization=True,
            )
            loss.backward()

            with torch.no_grad():
                self.gaussian_model.max_radii2D[vis_filter] = torch.max(
                    self.gaussian_model.max_radii2D[vis_filter], radii[vis_filter],
                )
                self.gaussian_model.add_densification_stats(viewspace_pts, vis_filter)

                if it % self.init_gaussian_update == 0:
                    self.gaussian_model.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self._iteration_count == self.init_gaussian_reset or (
                    self._iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussian_model.reset_opacity()

                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)

        self._occ_aware_visibility[uid] = (render_pkg["n_touched"] > 0).long()
        self._initialized = True

    def _add_new_keyframe(self, uid: int, depth_np: np.ndarray) -> np.ndarray:
        """Prepare depth map for keyframe insertion (RGB-D mode)."""
        viewpoint = self._cameras[uid]
        rgb_boundary_threshold = self.config["Training"].get("rgb_boundary_threshold", 0.01)
        gt_img = viewpoint.original_image.to(self.device)
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        initial_depth = torch.from_numpy(depth_np).unsqueeze(0).float()
        initial_depth[~valid_rgb.cpu()] = 0
        return initial_depth[0].numpy()

    # ─────────────────────────────────────────────────────────────────────────
    # Private – Tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _tracking(self, uid: int, viewpoint: Camera) -> None:
        """Optimise camera pose given current Gaussian map."""
        # Initialise from previous frame's pose
        prev_uid = uid - 1
        if prev_uid in self._cameras:
            prev = self._cameras[prev_uid]
            viewpoint.update_RT(prev.R, prev.T)

        opt_params = [
            {"params": [viewpoint.cam_rot_delta],
             "lr": self.config["Training"]["lr"]["cam_rot_delta"],
             "name": f"rot_{uid}"},
            {"params": [viewpoint.cam_trans_delta],
             "lr": self.config["Training"]["lr"]["cam_trans_delta"],
             "name": f"trans_{uid}"},
            {"params": [viewpoint.exposure_a], "lr": 0.01, "name": f"exp_a_{uid}"},
            {"params": [viewpoint.exposure_b], "lr": 0.01, "name": f"exp_b_{uid}"},
        ]
        pose_optimizer = torch.optim.Adam(opt_params)

        self._cur_visibility = None

        for _ in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussian_model, self.rasterizer_pipe, self.background,
            )
            if render_pkg is None:
                break

            image = render_pkg["render"]
            depth_r = render_pkg["depth"]
            opacity = render_pkg["opacity"]

            pose_optimizer.zero_grad()
            loss = get_loss_tracking(self.config, image, depth_r, opacity, viewpoint)
            loss.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                break

        self._median_depth = get_median_depth(
            render_pkg["depth"], render_pkg["opacity"],
        )
        self._cur_visibility = (render_pkg["n_touched"] > 0).long()
        self._current_camera = viewpoint
        self._current_pose = self._camera_to_w2c(viewpoint)

    # ─────────────────────────────────────────────────────────────────────────
    # Private – Keyframe decision
    # ─────────────────────────────────────────────────────────────────────────

    def _is_keyframe(self, uid: int, last_kf_vis, cur_vis) -> bool:
        if self._last_kf_idx == uid:
            return False
        if last_kf_vis is None or cur_vis is None:
            return True

        curr = self._cameras[uid]
        last_kf = self._cameras[self._last_kf_idx]

        pose_CW = getWorld2View2(curr.R, curr.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])

        dist_check  = dist > self.kf_translation * self._median_depth
        dist_check2 = dist > self.kf_min_translation * self._median_depth

        min_len = min(len(cur_vis), len(last_kf_vis))
        cur_v = cur_vis[:min_len]
        last_v = last_kf_vis[:min_len]
        union = torch.logical_or(cur_v, last_v).count_nonzero()
        intersection = torch.logical_and(cur_v, last_v).count_nonzero()
        if union == 0:
            return True
        point_ratio = intersection / union

        return (point_ratio < self.kf_overlap and dist_check2) or dist_check

    # ─────────────────────────────────────────────────────────────────────────
    # Private – Mapping
    # ─────────────────────────────────────────────────────────────────────────

    def _mapping(self, current_window: list, iters: int = 1) -> None:
        """Run mapping iterations over the keyframe window."""
        if not current_window:
            return

        viewpoint_stack = [self._cameras[idx] for idx in current_window
                           if idx in self._cameras]
        if not viewpoint_stack:
            return

        for _ in range(iters):
            self._iteration_count += 1

            loss_mapping = torch.tensor(0.0, device=self.device)
            viewspace_acm, vis_acm, radii_acm, n_touched_acm = [], [], [], []

            for viewpoint in viewpoint_stack:
                render_pkg = render(
                    viewpoint, self.gaussian_model,
                    self.rasterizer_pipe, self.background,
                )
                if render_pkg is None:
                    continue

                image = render_pkg["render"]
                depth_r = render_pkg["depth"]
                opacity = render_pkg["opacity"]
                viewspace_pts = render_pkg["viewspace_points"]
                vis_filter = render_pkg["visibility_filter"]
                radii = render_pkg["radii"]
                n_touched = render_pkg["n_touched"]

                loss_mapping = loss_mapping + get_loss_mapping(
                    self.config, image, depth_r, viewpoint, opacity,
                )
                viewspace_acm.append(viewspace_pts)
                vis_acm.append(vis_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            # Isotropic regularisation (same as MonoGS)
            scaling = self.gaussian_model.get_scaling
            if scaling.shape[0] > 0:
                iso_loss = torch.abs(scaling - scaling.mean(dim=1, keepdim=True))
                loss_mapping = loss_mapping + 10 * iso_loss.mean()

            loss_mapping.backward()

            # Accumulate gradient-based FIM (diagonal Hessian approx).
            # Λ_k += (∂L/∂θ_k)² per Gaussian parameter.
            self._accumulate_fim()

            with torch.no_grad():
                # Update occ-aware visibility
                for i, idx in enumerate(current_window):
                    if i < len(n_touched_acm):
                        self._occ_aware_visibility[idx] = (n_touched_acm[i] > 0).long()

                # Densification stats
                for i in range(len(viewspace_acm)):
                    self.gaussian_model.max_radii2D[vis_acm[i]] = torch.max(
                        self.gaussian_model.max_radii2D[vis_acm[i]],
                        radii_acm[i][vis_acm[i]],
                    )
                    self.gaussian_model.add_densification_stats(
                        viewspace_acm[i], vis_acm[i],
                    )

                update_gaussian = (
                    self._iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussian_model.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    # Hard cap to prevent OOM on multi-agent runs
                    N = self.gaussian_model._xyz.shape[0]
                    if N > self.max_gaussians:
                        opac = self.gaussian_model.get_opacity.squeeze(-1)
                        _, keep_idx = opac.topk(self.max_gaussians)
                        keep = torch.zeros(N, dtype=torch.bool, device=opac.device)
                        keep[keep_idx] = True
                        self.gaussian_model.prune_points(~keep)

                if (self._iteration_count % self.gaussian_reset) == 0 and not update_gaussian:
                    self.gaussian_model.reset_opacity_nonvisible(vis_acm)

                self.gaussian_model.optimizer.step()
                self.gaussian_model.optimizer.zero_grad(set_to_none=True)
                self.gaussian_model.update_learning_rate(self._iteration_count)

    # ─────────────────────────────────────────────────────────────────────────
    # Private – Pose helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _accumulate_fim(self) -> None:
        """Accumulate squared gradients as FIM diagonal approximation."""
        gm = self.gaussian_model
        N = gm._xyz.shape[0]
        if N == 0:
            return

        def _sq_grad(param):
            if param.grad is not None:
                return param.grad.detach().pow(2)
            return torch.zeros_like(param.data)

        g_means  = _sq_grad(gm._xyz)
        g_quats  = _sq_grad(gm._rotation)
        g_scales = _sq_grad(gm._scaling)
        g_opac   = _sq_grad(gm._opacity)

        if self._fim_means is not None and self._fim_means.shape[0] == N:
            self._fim_means  = self._fim_means + g_means
            self._fim_quats  = self._fim_quats + g_quats
            self._fim_scales = self._fim_scales + g_scales
            self._fim_opac   = self._fim_opac + g_opac
        else:
            # Reset on shape change (densification/pruning changed N)
            self._fim_means  = g_means
            self._fim_quats  = g_quats
            self._fim_scales = g_scales
            self._fim_opac   = g_opac

    @staticmethod
    def _camera_to_w2c(camera: Camera) -> torch.Tensor:
        """Return [4,4] world-to-camera from Camera.R and Camera.T."""
        T = torch.eye(4, device=camera.R.device)
        T[:3, :3] = camera.R
        T[:3, 3]  = camera.T
        return T

    # ─────────────────────────────────────────────────────────────────────────
    # Post-experiment evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_render_metrics(self, max_keyframes: int = 20) -> dict:
        """
        Evaluate PSNR, SSIM, LPIPS, DepthL1 on retained keyframes.

        Only keyframes still have original_image/depth (non-keyframes are cleaned).
        Returns dict with mean metrics across sampled keyframes.
        """
        from pytorch_msssim import ms_ssim
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)

        kf_ids = self._kf_indices[-max_keyframes:]
        psnrs, ssims, lpipses, depth_l1s = [], [], [], []

        with torch.no_grad():
            for kid in kf_ids:
                vp = self._cameras.get(kid)
                if vp is None or vp.original_image is None:
                    continue

                pkg = render(vp, self.gaussian_model, self.rasterizer_pipe, self.background)
                if pkg is None:
                    continue

                rendered = pkg["render"].clamp(0, 1)        # [3, H, W]
                gt = vp.original_image.cuda().clamp(0, 1)   # [3, H, W]

                # PSNR
                mse = (rendered - gt).pow(2).mean().item()
                psnr = -10 * np.log10(max(mse, 1e-10))
                psnrs.append(psnr)

                # MS-SSIM (needs [B,C,H,W], min size 160)
                h, w = gt.shape[1], gt.shape[2]
                if min(h, w) >= 160:
                    ssim_val = ms_ssim(
                        rendered.unsqueeze(0), gt.unsqueeze(0),
                        data_range=1.0, size_average=True,
                    ).item()
                    ssims.append(ssim_val)

                # LPIPS (needs [B,C,H,W] in [-1,1])
                lpips_val = lpips_fn(
                    rendered.unsqueeze(0) * 2 - 1,
                    gt.unsqueeze(0) * 2 - 1,
                ).item()
                lpipses.append(lpips_val)

                # Depth L1
                depth_r = pkg["depth"]  # [1, H, W]
                if vp.depth is not None:
                    gt_d = torch.from_numpy(vp.depth).to(self.device).unsqueeze(0)
                    mask = gt_d > 0.01
                    if mask.any():
                        dl1 = (depth_r[mask] - gt_d[mask]).abs().mean().item()
                        depth_l1s.append(dl1)

        return {
            "PSNR":    float(np.mean(psnrs))    if psnrs    else float("nan"),
            "SSIM":    float(np.mean(ssims))    if ssims    else float("nan"),
            "LPIPS":   float(np.mean(lpipses))  if lpipses  else float("nan"),
            "DepthL1": float(np.mean(depth_l1s)) if depth_l1s else float("nan"),
        }

    def refine_map(self, iters: int = 3000) -> None:
        """Run additional mapping iterations on the keyframe window after experiment."""
        window = self._current_window[:self.window_size]
        if not window:
            window = self._kf_indices[-self.window_size:]
        if not window:
            return
        self._mapping(window, iters=iters)
