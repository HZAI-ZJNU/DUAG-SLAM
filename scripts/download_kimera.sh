#!/bin/bash
# Download Kimera-Multi Campus-Outdoor dataset.
# SOURCE: https://github.com/MIT-SPARK/Kimera-Multi  — check Releases tab.
set -e
mkdir -p data/kimera_campus

echo "=== Kimera-Multi Campus download ==="
echo "Visit: https://github.com/MIT-SPARK/Kimera-Multi"
echo "Go to Releases tab and download the Campus-Outdoor sequences."
echo "Each sequence contains RGB-D + IMU + RTK GPS for 6 robots."
echo ""
echo "Expected structure after download:"
echo "  data/kimera_campus/"
echo "    campus_outdoor_00/"
echo "      robot_0/  robot_1/  ...  robot_5/"
echo "      Each robot: rgb/, depth/, imu.csv, gps.csv, groundtruth.txt"
echo ""
echo "After download: ls data/kimera_campus/"
# ----- INSERT VERIFIED DOWNLOAD COMMAND FROM KIMERA-MULTI RELEASES -----
