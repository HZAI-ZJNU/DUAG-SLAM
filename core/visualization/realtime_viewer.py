# core/visualization/realtime_viewer.py
"""
Real-time visualization for DUAG-SLAM multi-agent experiments.

Shows a live window with:
  - Top row:    Input RGB for each agent
  - Bottom row: Gaussian-splatted render for each agent
  - Right panel: Top-down trajectory plot (GT dashed, estimated solid)

Usage:
    viewer = RealtimeViewer(n_agents=2, panel_h=340, panel_w=600)
    # in frame loop:
    viewer.update(agent_id, input_rgb_np, rendered_rgb_np, est_pose_4x4, gt_pose_4x4)
    viewer.show()  # cv2.imshow + waitKey
    # at end:
    viewer.close()
"""

import cv2
import numpy as np
from typing import Optional, List


class RealtimeViewer:
    """OpenCV-based real-time multi-agent SLAM visualizer."""

    AGENT_COLORS = [
        (66, 133, 244),   # blue  (BGR)
        (219, 68, 55),    # red
        (244, 180, 0),    # yellow
        (15, 157, 88),    # green
        (171, 71, 188),   # purple
        (255, 112, 67),   # orange
    ]
    GT_COLORS = [
        (180, 200, 255),  # light blue
        (200, 170, 170),  # light red
        (255, 230, 170),  # light yellow
        (170, 220, 180),  # light green
        (220, 190, 230),  # light purple
        (255, 200, 180),  # light orange
    ]

    def __init__(
        self,
        n_agents: int = 2,
        panel_h: int = 340,
        panel_w: int = 600,
        traj_size: int = 400,
        window_name: str = "DUAG-SLAM Live",
        save_video: Optional[str] = None,
        fps: int = 10,
    ):
        self.n_agents = n_agents
        self.panel_h = panel_h
        self.panel_w = panel_w
        self.traj_size = traj_size
        self.window_name = window_name

        # Canvas: 2 rows (input / rendered) × n_agents columns + trajectory panel
        self.canvas_h = panel_h * 2
        self.canvas_w = panel_w * n_agents + traj_size
        self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # Per-agent image slots (initialized to dark gray)
        self._input_panels = [
            np.full((panel_h, panel_w, 3), 40, dtype=np.uint8) for _ in range(n_agents)
        ]
        self._render_panels = [
            np.full((panel_h, panel_w, 3), 40, dtype=np.uint8) for _ in range(n_agents)
        ]

        # Trajectory history: list of (x, z) positions per agent
        self._est_traj: List[List[tuple]] = [[] for _ in range(n_agents)]
        self._gt_traj: List[List[tuple]] = [[] for _ in range(n_agents)]

        # Trajectory bounds (auto-scaled)
        self._traj_min = np.array([1e9, 1e9])
        self._traj_max = np.array([-1e9, -1e9])

        self._frame_idx = 0
        self._n_gaussians = [0] * n_agents

        # Video writer
        self._video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                save_video, fourcc, fps, (self.canvas_w, self.canvas_h)
            )

    def update(
        self,
        agent_id: int,
        input_rgb: Optional[np.ndarray],
        rendered_rgb: Optional[np.ndarray],
        est_pose: Optional[np.ndarray] = None,
        gt_pose: Optional[np.ndarray] = None,
        n_gaussians: int = 0,
    ) -> None:
        """
        Update display data for one agent.

        Args:
            agent_id: 0-indexed agent ID
            input_rgb: [H, W, 3] uint8 or float32 [0,1] input image
            rendered_rgb: [H, W, 3] uint8 or float32 [0,1] Gaussian render
            est_pose: [4, 4] estimated camera-to-world pose
            gt_pose: [4, 4] ground-truth camera-to-world pose
            n_gaussians: current Gaussian count for this agent
        """
        if input_rgb is not None:
            self._input_panels[agent_id] = self._prep_image(input_rgb)
        if rendered_rgb is not None:
            self._render_panels[agent_id] = self._prep_image(rendered_rgb)

        if est_pose is not None:
            pos = est_pose[:3, 3]
            self._est_traj[agent_id].append((pos[0], pos[2]))  # x, z (top-down)
            self._update_bounds(pos[0], pos[2])

        if gt_pose is not None:
            pos = gt_pose[:3, 3]
            self._gt_traj[agent_id].append((pos[0], pos[2]))
            self._update_bounds(pos[0], pos[2])

        self._n_gaussians[agent_id] = n_gaussians

    def show(self, wait_ms: int = 1) -> bool:
        """
        Compose and display the canvas. Returns False if user pressed 'q'.
        """
        self._frame_idx += 1

        # Fill image panels
        for i in range(self.n_agents):
            y0 = 0
            x0 = i * self.panel_w
            self.canvas[y0:y0 + self.panel_h, x0:x0 + self.panel_w] = self._input_panels[i]
            y1 = self.panel_h
            self.canvas[y1:y1 + self.panel_h, x0:x0 + self.panel_w] = self._render_panels[i]

            # Labels
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            cv2.putText(self.canvas, f"Agent {i} - Input",
                        (x0 + 10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(self.canvas, f"Agent {i} - Rendered ({self._n_gaussians[i]} G)",
                        (x0 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw trajectory panel
        traj_panel = self._draw_trajectory()
        tx0 = self.panel_w * self.n_agents
        self.canvas[:self.canvas_h, tx0:tx0 + self.traj_size] = traj_panel

        # Frame counter
        cv2.putText(self.canvas, f"Frame {self._frame_idx}",
                    (tx0 + 10, self.canvas_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show
        cv2.imshow(self.window_name, self.canvas)

        # Write video frame
        if self._video_writer is not None:
            self._video_writer.write(self.canvas)

        key = cv2.waitKey(wait_ms) & 0xFF
        return key != ord("q")

    def close(self) -> None:
        """Clean up windows and video writer."""
        if self._video_writer is not None:
            self._video_writer.release()
        cv2.destroyAllWindows()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _prep_image(self, img: np.ndarray) -> np.ndarray:
        """Convert to uint8 BGR at panel size."""
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.resize(img, (self.panel_w, self.panel_h))

    def _update_bounds(self, x: float, z: float) -> None:
        margin = 0.5
        self._traj_min[0] = min(self._traj_min[0], x - margin)
        self._traj_min[1] = min(self._traj_min[1], z - margin)
        self._traj_max[0] = max(self._traj_max[0], x + margin)
        self._traj_max[1] = max(self._traj_max[1], z + margin)

    def _world_to_pixel(self, x: float, z: float) -> tuple:
        """Map world (x, z) to pixel coords in the trajectory panel."""
        pad = 30
        rng = self._traj_max - self._traj_min
        rng = np.maximum(rng, 1e-3)
        usable = self.traj_size - 2 * pad
        px = int(pad + (x - self._traj_min[0]) / rng[0] * usable)
        pz = int(pad + (z - self._traj_min[1]) / rng[1] * usable)
        # Flip z so "forward" goes up
        pz = self.canvas_h - pz
        return (px, pz)

    def _draw_trajectory(self) -> np.ndarray:
        """Render top-down trajectory panel."""
        panel = np.full((self.canvas_h, self.traj_size, 3), 30, dtype=np.uint8)

        # Title
        cv2.putText(panel, "Top-Down Trajectories",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        for i in range(self.n_agents):
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            gt_color = self.GT_COLORS[i % len(self.GT_COLORS)]

            # Draw GT trajectory (dashed effect via dotted line)
            gt = self._gt_traj[i]
            if len(gt) > 1:
                pts = [self._world_to_pixel(x, z) for x, z in gt]
                for j in range(0, len(pts) - 1, 2):  # dotted
                    cv2.line(panel, pts[j], pts[min(j + 1, len(pts) - 1)], gt_color, 1)

            # Draw estimated trajectory (solid)
            est = self._est_traj[i]
            if len(est) > 1:
                pts = [self._world_to_pixel(x, z) for x, z in est]
                for j in range(len(pts) - 1):
                    cv2.line(panel, pts[j], pts[j + 1], color, 2)

            # Current position marker
            if est:
                px, pz = self._world_to_pixel(*est[-1])
                cv2.circle(panel, (px, pz), 5, color, -1)
                cv2.putText(panel, f"A{i}", (px + 8, pz + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Legend
        y_leg = self.canvas_h - 60
        cv2.putText(panel, "--- GT", (10, y_leg), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(panel, "--- Est", (10, y_leg + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

        return panel
