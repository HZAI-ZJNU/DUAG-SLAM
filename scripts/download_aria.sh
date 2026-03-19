#!/bin/bash
# Download Aria-Multiagent dataset.
# SOURCE: repos/MAGiC-SLAM/README.md AND repos/MAC-Ego3D/README.md
set -e
mkdir -p data/aria_multiagent

echo "=== Checking MAGiC-SLAM README ==="
grep -i "aria\|download\|dataset\|wget\|gdown" repos/MAGiC-SLAM/README.md | head -20
echo ""
echo "=== Checking MAC-Ego3D README ==="
grep -i "aria\|download\|dataset\|wget\|gdown" repos/MAC-Ego3D/README.md | head -20
echo ""
echo "INSERT verified download command from READMEs above."
# ----- INSERT VERIFIED COMMAND HERE -----
