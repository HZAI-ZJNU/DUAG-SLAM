#!/bin/bash
# Download SubT-MRS dataset (fallback if CSA Mars Yard unavailable).
# SOURCE: https://superodometry.com/datasets
# Also check: https://theairlab.org/dataset/interestingness
set -e
mkdir -p data/subtmrs

echo "=== SubT-MRS download ==="
echo "Visit: https://superodometry.com/datasets"
echo "Download the multi-robot sequences (4 robots, underground tunnel)."
echo "Select sequences with: RGB-D, LiDAR, IMU for all robots."
echo ""
echo "If CSA Mars Yard (IEEE DataPort, Lajoie 2026) is available, use that instead."
echo "Check: https://ieee-dataport.org/  search 'CSA Mars Yard SLAM'"
echo ""
echo "Expected structure after download:"
echo "  data/subtmrs/"
echo "    sequence_01/"
echo "      robot_0/  robot_1/  robot_2/  robot_3/"
echo "      Each: rgb/, depth/, lidar/, groundtruth.txt"
# ----- INSERT VERIFIED DOWNLOAD COMMAND -----
