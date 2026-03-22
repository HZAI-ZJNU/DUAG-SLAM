#!/usr/bin/env python3
"""
Compute stereo depth maps from S3Ev1 left+right image pairs.

Uses OpenCV StereoSGBM with calibration from alpha/bob/carol.yaml.
Output: data/s3e/S3Ev1/{sequence}/{Robot}/Camera/depth/{timestamp}.png
        (16-bit PNG, depth in mm, depth_scale=1000)

Usage:
    python scripts/compute_s3e_stereo_depth.py [--sequences SEQ1 SEQ2 ...] [--max-frames N]
"""
import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

DATA_ROOT = Path("data/s3e/S3Ev1")
CALIB_DIR = DATA_ROOT / "Calibration"
ROBOTS = ["Alpha", "Bob", "Carol"]
SEQUENCES = ["S3E_Playground_2", "S3E_Square_2"]  # Laboratory_1 excluded (no usable GT)


def parse_opencv_matrix(yaml_text: str, key: str) -> np.ndarray:
    """Parse an !!opencv-matrix from a YAML string."""
    # Find the block starting with key
    pattern = rf'{re.escape(key)}:\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[([\d\s.,eE+\-]+)\]'
    match = re.search(pattern, yaml_text, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot find opencv-matrix '{key}' in YAML")
    rows, cols = int(match.group(1)), int(match.group(2))
    data = [float(x.strip()) for x in match.group(3).split(',')]
    return np.array(data, dtype=np.float64).reshape(rows, cols)


def load_stereo_calibration(robot_name: str):
    """Load stereo calibration for a given robot from its YAML file."""
    calib_path = CALIB_DIR / f"{robot_name.lower()}.yaml"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration not found: {calib_path}")

    text = calib_path.read_text()

    K_left = parse_opencv_matrix(text, "LEFT.K")
    D_left = parse_opencv_matrix(text, "LEFT.D")
    R_left = parse_opencv_matrix(text, "LEFT.R")
    P_left = parse_opencv_matrix(text, "LEFT.P")

    K_right = parse_opencv_matrix(text, "RIGHT.K")
    D_right = parse_opencv_matrix(text, "RIGHT.D")
    R_right = parse_opencv_matrix(text, "RIGHT.R")
    P_right = parse_opencv_matrix(text, "RIGHT.P")

    size = (int(re.search(r'LEFT\.width:\s*(\d+)', text).group(1)),
            int(re.search(r'LEFT\.height:\s*(\d+)', text).group(1)))

    # bf = baseline × fx (from ORB-SLAM config section)
    bf_match = re.search(r'Camera\.bf:\s*([\d.]+)', text)
    bf = float(bf_match.group(1)) if bf_match else abs(P_right[0, 3])

    return {
        "K_left": K_left, "D_left": D_left, "R_left": R_left, "P_left": P_left,
        "K_right": K_right, "D_right": D_right, "R_right": R_right, "P_right": P_right,
        "size": size,  # (width, height)
        "bf": bf,
    }


def create_rectify_maps(calib):
    """Create undistort+rectify maps for left and right cameras."""
    w, h = calib["size"]
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        calib["K_left"], calib["D_left"], calib["R_left"], calib["P_left"],
        (w, h), cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        calib["K_right"], calib["D_right"], calib["R_right"], calib["P_right"],
        (w, h), cv2.CV_32FC1)
    return (map1_left, map2_left), (map1_right, map2_right)


def compute_depth_map(img_left, img_right, maps_left, maps_right, bf, sgbm):
    """Compute depth from stereo pair. Returns depth in meters as float32."""
    # Rectify
    rect_left = cv2.remap(img_left, maps_left[0], maps_left[1], cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, maps_right[0], maps_right[1], cv2.INTER_LINEAR)

    # Convert to grayscale for SGBM
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity (returns fixed-point: divide by 16)
    disparity = sgbm.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Convert to depth: depth = bf / disparity
    # Invalid disparities (<=0) get depth=0
    valid = disparity > 0.5
    depth = np.zeros_like(disparity)
    depth[valid] = bf / disparity[valid]

    # Clamp to reasonable range (0.1m - 50m)
    depth = np.clip(depth, 0, 50.0)
    depth[~valid] = 0.0

    return depth


def process_robot(seq_name: str, robot_name: str, calib, max_frames: int = -1):
    """Process all stereo pairs for one robot in one sequence."""
    left_dir = DATA_ROOT / seq_name / robot_name / "Camera" / "image_left"
    right_dir = DATA_ROOT / seq_name / robot_name / "Camera" / "image_right"
    depth_dir = DATA_ROOT / seq_name / robot_name / "Camera" / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    left_files = sorted(left_dir.glob("*.jpg"))
    if max_frames > 0:
        left_files = left_files[:max_frames]

    if not left_files:
        print(f"  No images found in {left_dir}")
        return 0

    # Build right-image timestamp index (left/right timestamps differ by ~2ms)
    right_files = sorted(right_dir.glob("*.jpg"))
    right_timestamps = np.array([float(p.stem) for p in right_files])
    right_paths_arr = right_files

    # Create rectification maps (once per robot)
    maps_left, maps_right = create_rectify_maps(calib)

    # StereoSGBM parameters tuned for 1224×1024 indoor/outdoor scenes
    min_disp = 0
    num_disp = 128  # must be divisible by 16
    block_size = 7
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    count = 0
    for i, left_path in enumerate(left_files):
        ts_name = left_path.stem  # e.g. "1661164286.638999939"
        depth_path = depth_dir / f"{ts_name}.png"

        if depth_path.exists():
            count += 1
            continue  # skip already computed

        # Find nearest right image by timestamp
        left_ts = float(ts_name)
        idx = np.argmin(np.abs(right_timestamps - left_ts))
        if abs(right_timestamps[idx] - left_ts) > 0.05:  # max 50ms tolerance
            continue
        right_path = right_paths_arr[idx]

        img_left = cv2.imread(str(left_path))
        img_right = cv2.imread(str(right_path))
        if img_left is None or img_right is None:
            continue

        depth = compute_depth_map(
            img_left, img_right, maps_left, maps_right, calib["bf"], sgbm)

        # Save as 16-bit PNG (millimeters)
        depth_mm = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)
        count += 1

        if (i + 1) % 200 == 0:
            print(f"    {robot_name}: {i+1}/{len(left_files)} frames processed")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs="*", default=SEQUENCES)
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    # Load calibration (same for all robots in S3Ev1)
    for seq in args.sequences:
        print(f"\n=== {seq} ===")
        for robot in ROBOTS:
            calib = load_stereo_calibration(robot)
            print(f"  Processing {robot} (bf={calib['bf']:.2f}, size={calib['size']})...")
            n = process_robot(seq, robot, calib, args.max_frames)
            print(f"  {robot}: {n} depth maps computed/found")


if __name__ == "__main__":
    main()
