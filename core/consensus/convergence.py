# core/consensus/convergence.py
# Tracks ADMM convergence history. Used for early stopping and for Figure 5 in thesis.

from collections import deque
from typing import List
import numpy as np


class ConvergenceMonitor:
    def __init__(self, window_size: int = 10, diverge_factor: float = 100.0):
        self.primal_history: deque       = deque(maxlen=window_size)
        self.dual_history:   deque       = deque(maxlen=window_size)
        self.rho_history:    List[float] = []
        self.diverge_factor = diverge_factor

    def update(self, primal_res: float, dual_res: float, rho: float) -> None:
        self.primal_history.append(primal_res)
        self.dual_history.append(dual_res)
        self.rho_history.append(rho)

    def is_diverging(self) -> bool:
        if len(self.primal_history) < 5:
            return False
        return self.primal_history[-1] > self.diverge_factor * self.primal_history[0]

    def convergence_rate(self) -> float:
        """Log-linear slope of primal residuals. Negative = converging."""
        if len(self.primal_history) < 3:
            return float('nan')
        log_r = np.log([max(r, 1e-10) for r in self.primal_history])
        return float(np.polyfit(range(len(log_r)), log_r, 1)[0])

    def summary(self) -> dict:
        return {
            "last_primal":      self.primal_history[-1] if self.primal_history else None,
            "last_dual":        self.dual_history[-1]   if self.dual_history   else None,
            "last_rho":         self.rho_history[-1]    if self.rho_history    else None,
            "convergence_rate": self.convergence_rate(),
            "n_iterations":     len(self.rho_history),
        }
