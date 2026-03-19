#!/usr/bin/env python3
"""
Compute Expected Calibration Error (ECE) for the Aria-Multiagent experiment.
Tests H3: does our Hessian FIM correctly identify unreliable Gaussians?

ECE measures whether high-predicted-uncertainty Gaussians actually have
higher reconstruction error than low-uncertainty Gaussians.

Usage: python scripts/compute_ece.py --result_dir outputs/results/DUAG-C_Aria-Multiagent/
"""
import argparse
import os

import numpy as np


def compute_ece(
    fim_traces: np.ndarray,    # [N] predicted uncertainty (lower FIM = more uncertain)
    recon_errors: np.ndarray,  # [N] actual per-Gaussian reconstruction error
    n_bins: int = 10,
) -> float:
    """
    ECE = sum_{bin b} |bin_size/N| * |mean_error(b) - mean_predicted_uncertainty(b)|

    Well-calibrated: ECE < 0.05
    """
    # Normalize both to [0,1] for meaningful comparison
    uncertainty = 1.0 / np.sqrt(fim_traces + 1e-10)
    ptp = uncertainty.ptp()
    uncertainty = (uncertainty - uncertainty.min()) / (ptp + 1e-10) if ptp > 0 else uncertainty * 0
    ptp_e = recon_errors.ptp()
    recon_norm = (recon_errors - recon_errors.min()) / (ptp_e + 1e-10) if ptp_e > 0 else recon_errors * 0

    N = len(uncertainty)
    bin_idx = np.floor(uncertainty * n_bins).clip(0, n_bins - 1).astype(int)
    ece = 0.0

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        pred_mean = uncertainty[mask].mean()
        error_mean = recon_norm[mask].mean()
        ece += (mask.sum() / N) * abs(error_mean - pred_mean)

    return float(ece)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", default="outputs/results/DUAG-C_Aria-Multiagent")
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    # Load FIM traces and reconstruction errors from saved result files
    fim_file = f"{args.result_dir}/scene_01_robot_0_fim_traces.npy"
    error_file = f"{args.result_dir}/scene_01_robot_0_recon_errors.npy"

    if not (os.path.exists(fim_file) and os.path.exists(error_file)):
        print(f"Files not found: {fim_file}, {error_file}")
        print("Ensure aria_multiagent_duag_c.yaml has:")
        print("  system.save_fim_traces: true")
        print("  system.save_recon_errors: true")
        print("Then re-run the Aria experiment (Step 19).")
        return

    fim_traces = np.load(fim_file)
    recon_errors = np.load(error_file)
    ece = compute_ece(fim_traces, recon_errors, args.n_bins)

    print(f"ECE (Expected Calibration Error): {ece:.4f}")
    print(f"Threshold for H3 validation: ECE < 0.05")
    print(f"H3 VALIDATED: {ece < 0.05}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Calibration plot
        uncertainty = 1.0 / np.sqrt(fim_traces + 1e-10)
        ptp = uncertainty.ptp()
        uncertainty = (uncertainty - uncertainty.min()) / (ptp + 1e-10) if ptp > 0 else uncertainty * 0
        ptp_e = recon_errors.ptp()
        recon_norm = (recon_errors - recon_errors.min()) / (ptp_e + 1e-10) if ptp_e > 0 else recon_errors * 0
        n_bins = args.n_bins
        bin_idx = np.floor(uncertainty * n_bins).clip(0, n_bins - 1).astype(int)
        pred_means = [uncertainty[bin_idx == b].mean() for b in range(n_bins) if (bin_idx == b).any()]
        err_means = [recon_norm[bin_idx == b].mean() for b in range(n_bins) if (bin_idx == b).any()]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.scatter(pred_means, err_means, c='blue', label=f'DUAG-C (ECE={ece:.3f})')
        ax.set_xlabel("Predicted uncertainty (normalized)")
        ax.set_ylabel("Actual error (normalized)")
        ax.set_title("Uncertainty Calibration Plot (H3)")
        ax.legend()
        os.makedirs("outputs/figures", exist_ok=True)
        fig.savefig("outputs/figures/h3_calibration.pdf", dpi=150)
        print("Calibration plot saved: outputs/figures/h3_calibration.pdf")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
