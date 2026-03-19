#!/usr/bin/env python3
"""
Render a DUAG-SLAM merged Gaussian map to a PNG for paper Figure 6.
Usage: python scripts/visualize_map.py --ply outputs/maps/<exp>/merged_map.ply
"""
import argparse
import os

import numpy as np


def render_ply_to_image(ply_path: str, output_path: str, viewpoint: str = "front") -> None:
    """
    Renders a .ply point cloud (Gaussian map exported as points) to a PNG.
    Uses Open3D for headless rendering. Falls back to matplotlib scatter if unavailable.
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=600)
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        if viewpoint == "top":
            ctr.set_up([0, 0, 1])
            ctr.set_front([0, -1, 0])
        else:
            ctr.set_up([0, 1, 0])
            ctr.set_front([0, 0, -1])
        ctr.set_zoom(0.5)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()
        print(f"Rendered to: {output_path}")
    except ImportError:
        # Fallback: matplotlib scatter plot of point positions
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Parse PLY manually (ASCII format)
            points = []
            in_data = False
            vertex_count = 0
            with open(ply_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("element vertex"):
                        vertex_count = int(line.split()[-1])
                    elif line == "end_header":
                        in_data = True
                        continue
                    elif in_data and len(points) < vertex_count:
                        parts = line.split()
                        if len(parts) >= 3:
                            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

            if not points:
                print(f"No points found in {ply_path}")
                return

            pts = np.array(points)
            # Subsample if too many points
            step = max(1, len(pts) // 10000)
            pts = pts[::step]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.1, c='steelblue', alpha=0.5)
            ax.set_title(os.path.basename(ply_path))
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"Fallback scatter rendered to: {output_path}")
        except ImportError:
            print("Install open3d or matplotlib for rendering.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--view", default="front", choices=["front", "top", "side"])
    args = parser.parse_args()
    out = args.output or args.ply.replace(".ply", f"_{args.view}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    render_ply_to_image(args.ply, out, args.view)


if __name__ == "__main__":
    main()
