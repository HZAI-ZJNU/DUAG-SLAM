#!/bin/bash
# Download Replica-Multiagent dataset for DUAG-C evaluation.
# SOURCE: Read repos/MAGiC-SLAM/README.md for the exact download command.
# The dataset is hosted by the MAGiC-SLAM authors.
# After reading the README, insert the verified download command below.
set -e
mkdir -p data/replica_multiagent

echo "=== Checking repos/MAGiC-SLAM/README.md for download instructions ==="
grep -i "replica\|download\|dataset\|wget\|gdown" repos/MAGiC-SLAM/README.md | head -30

echo ""
echo "INSERT the verified download command from the README above, then re-run."
echo "Example pattern (verify first):"
echo "  cd data/replica_multiagent && gdown <google_drive_id>"
echo "  # or: wget <url>"
echo ""
echo "After download, verify with: ls data/replica_multiagent/"
# ----- INSERT VERIFIED COMMAND HERE -----
