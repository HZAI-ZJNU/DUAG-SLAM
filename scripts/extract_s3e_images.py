#!/usr/bin/env python3
"""
Extract stereo camera images from S3Ev1 ROS2 .db3 bags.

Based on the rosbag_extractor repo (github.com/thisparticle/rosbag_extractor):
- S3Ev1 CompressedImage topics: /{Robot}/left_camera/compressed
                                 /{Robot}/right_camera/compressed
- ComIMG2JPG uses cv_bridge to decode CompressedImage -> OpenCV Mat -> .jpg
- File naming: {timestamp_sec:.9f}.jpg

We replicate this in pure Python using the `rosbags` library (no ROS install needed).
CompressedImage.data contains raw JPEG bytes, so we can either:
  (a) write bytes directly to .jpg  (fastest, no decode)
  (b) decode via cv2.imdecode for validation

Output structure (mirrors rosbag_extractor conventions):
  data/s3e/S3Ev1/{sequence}/{robot}/Camera/image_left/{timestamp}.jpg
  data/s3e/S3Ev1/{sequence}/{robot}/Camera/image_right/{timestamp}.jpg

Usage:
  conda run -n duag python scripts/extract_s3e_images.py
  conda run -n duag python scripts/extract_s3e_images.py --sequences S3E_Playground_2
  conda run -n duag python scripts/extract_s3e_images.py --left-only   # skip right camera
"""

import os
import sys
import argparse
import sqlite3
from pathlib import Path

# rosbags for ROS2 bag deserialization
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores

# Initialize typestore for ROS2 Humble message definitions
typestore = get_typestore(Stores.ROS2_HUMBLE)


# S3Ev1 robot names and their topic prefixes
ROBOTS = {
    "Alpha": {"left": "/Alpha/left_camera/compressed",
              "right": "/Alpha/right_camera/compressed"},
    "Bob":   {"left": "/Bob/left_camera/compressed",
              "right": "/Bob/right_camera/compressed"},
    "Carol": {"left": "/Carol/left_camera/compressed",
              "right": "/Carol/right_camera/compressed"},
}

SEQUENCES = ["S3E_Playground_2", "S3E_Square_2", "S3E_Laboratory_1"]

DATA_ROOT = Path("data/s3e/S3Ev1")


def extract_sequence(seq_name: str, left_only: bool = False, max_frames: int = 0,
                     data_root: Path = DATA_ROOT):
    """Extract images from one sequence's .db3 bag."""
    bag_dir = data_root / seq_name
    db3_path = bag_dir / f"{seq_name}.db3"

    if not db3_path.exists():
        print(f"  SKIP: {db3_path} not found")
        return

    # We need a metadata.yaml next to the .db3 for rosbags Reader
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        print(f"  SKIP: {metadata_path} not found (needed by rosbags Reader)")
        return

    print(f"\n{'='*60}")
    print(f"  Extracting: {seq_name}")
    print(f"  Bag: {db3_path} ({db3_path.stat().st_size / 1e9:.1f} GB)")
    print(f"{'='*60}")

    # Build topic -> (robot, side) mapping
    topic_map = {}
    for robot_name, topics in ROBOTS.items():
        topic_map[topics["left"]] = (robot_name, "left")
        if not left_only:
            topic_map[topics["right"]] = (robot_name, "right")

    # Create output directories
    for robot_name in ROBOTS:
        out_left = bag_dir / robot_name / "Camera" / "image_left"
        out_left.mkdir(parents=True, exist_ok=True)
        if not left_only:
            out_right = bag_dir / robot_name / "Camera" / "image_right"
            out_right.mkdir(parents=True, exist_ok=True)

    # Count extracted per robot/side
    counts = {r: {"left": 0, "right": 0} for r in ROBOTS}

    with Reader(bag_dir) as reader:
        # Filter to only camera topics
        connections = [c for c in reader.connections if c.topic in topic_map]
        if not connections:
            print(f"  ERROR: No camera topics found in bag!")
            print(f"  Available topics: {[c.topic for c in reader.connections]}")
            return

        print(f"  Found {len(connections)} camera topic connections")
        for c in connections:
            print(f"    {c.topic} ({c.msgtype})")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            robot_name, side = topic_map[connection.topic]

            # Skip if this robot/side already hit max_frames
            if max_frames > 0 and counts[robot_name][side] >= max_frames:
                continue

            # Deserialize CompressedImage message
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            # msg.header.stamp has sec and nanosec fields
            ts_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            # File naming: match rosbag_extractor convention
            # fmt::format("{:0<10.9f}", timestamp) -> "1661164288.227422200"
            filename = f"{ts_sec:.9f}.jpg"

            # Output path
            side_folder = "image_left" if side == "left" else "image_right"
            out_path = bag_dir / robot_name / "Camera" / side_folder / filename

            # CompressedImage.data is raw JPEG/PNG bytes
            # Write directly — same as cv_bridge -> cv::imwrite but faster
            jpg_data = bytes(msg.data)
            with open(out_path, "wb") as f:
                f.write(jpg_data)

            counts[robot_name][side] += 1

            # Progress every 500 frames
            total = sum(counts[r][s] for r in counts for s in counts[r])
            if total % 500 == 0:
                print(f"    Extracted {total} images so far...")

            if max_frames > 0 and counts[robot_name][side] >= max_frames:
                # Check if all done
                if all(counts[r]["left"] >= max_frames for r in ROBOTS):
                    if left_only or all(counts[r]["right"] >= max_frames for r in ROBOTS):
                        break

    # Summary
    print(f"\n  Summary for {seq_name}:")
    for robot_name in ROBOTS:
        left_c = counts[robot_name]["left"]
        right_c = counts[robot_name]["right"]
        print(f"    {robot_name}: {left_c} left, {right_c} right")


def main():
    parser = argparse.ArgumentParser(description="Extract stereo images from S3Ev1 .db3 bags")
    parser.add_argument("--sequences", nargs="+", default=SEQUENCES,
                        help="Sequence names to extract")
    parser.add_argument("--left-only", action="store_true",
                        help="Only extract left camera (saves time/space)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per robot per side (0=all)")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT),
                        help="Root data directory")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    print(f"S3Ev1 Image Extractor")
    print(f"Data root: {data_root}")
    print(f"Sequences: {args.sequences}")
    print(f"Left only: {args.left_only}")
    if args.max_frames > 0:
        print(f"Max frames: {args.max_frames} per robot per side")

    for seq in args.sequences:
        extract_sequence(seq, left_only=args.left_only, max_frames=args.max_frames,
                         data_root=data_root)

    print(f"\nDone! Images saved under {data_root}/{{sequence}}/{{Robot}}/Camera/")


if __name__ == "__main__":
    main()
