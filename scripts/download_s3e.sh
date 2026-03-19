#!/bin/bash
# Download S3E dataset (3 UGVs, 18 sequences, LiDAR+stereo+IMU+UWB).
# SOURCE: github.com/PengYu-Team/S3E  — read their README carefully.
# DOI: https://doi.org/10.1145/3503161.3548194
set -e
mkdir -p data/s3e

# S3E is hosted on Google Drive via the PengYu-Team GitHub repo.
# Sequences available: outdoor_01..13, indoor_01..05
# Each sequence contains: robot_0/, robot_1/, robot_2/ subdirs
# Each robot dir: rgb/, depth/, imu.csv, groundtruth.txt (TUM format)

echo "=== S3E download ==="
echo "Visit: https://github.com/PengYu-Team/S3E"
echo "Read their README for the Google Drive download link."
echo "Download all sequences to data/s3e/"
echo ""
echo "Expected structure after download:"
echo "  data/s3e/"
echo "    outdoor_01/"
echo "      robot_0/rgb/, robot_0/depth/, robot_0/groundtruth.txt"
echo "      robot_1/rgb/, robot_1/depth/, robot_1/groundtruth.txt"
echo "      robot_2/rgb/, robot_2/depth/, robot_2/groundtruth.txt"
echo "    outdoor_02/  ..."
echo "    indoor_01/   ..."
echo ""
echo "After download: ls data/s3e/ to verify"
# ----- INSERT VERIFIED DOWNLOAD COMMAND FROM S3E README -----
