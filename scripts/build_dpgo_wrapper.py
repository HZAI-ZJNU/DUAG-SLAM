#!/usr/bin/env python3
"""scripts/build_dpgo_wrapper.py — Build the DPGO pybind11 wrapper."""
import subprocess, os, sys

DPGO_DIR = "repos/dpgo"
WRAPPER = "extracted/dpgo_wrapper"

assert os.path.isdir(DPGO_DIR), f"DPGO repo not found at {DPGO_DIR}"

# List DPGO headers
result = subprocess.run(["find", DPGO_DIR, "-name", "*.h"], capture_output=True, text=True)
print("=== DPGO headers ===")
for h in sorted(result.stdout.strip().split("\n")):
    print(f"  {h}")

# Build DPGO C++ library
build_dir = os.path.join(DPGO_DIR, "build")
os.makedirs(build_dir, exist_ok=True)
subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True)
subprocess.run(["make", f"-j{os.cpu_count()}"], cwd=build_dir, check=True)

# Build pybind11 wrapper
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=WRAPPER, check=True)

sys.path.insert(0, WRAPPER)
try:
    import dpgo_pybind
    print("SUCCESS: dpgo_pybind imported.")
except ImportError as e:
    print(f"FAILED: {e}. Update dpgo_pybind.cpp with verified DPGO API.")
    sys.exit(1)
