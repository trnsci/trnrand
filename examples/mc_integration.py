"""
Monte Carlo vs Quasi-Monte Carlo integration.

Compares pseudo-random (trnrand.uniform) vs quasi-random (trnrand.sobol)
for numerical integration. QMC typically converges O(1/N) vs O(1/√N)
for pseudo-random, which matters when each function evaluation is expensive
(e.g., electron integral evaluation in quantum chemistry).

Usage:
    python examples/mc_integration.py
"""

import time
import torch
import trnrand


def sphere_volume_mc(n_points: int, n_dims: int, quasi: bool = False) -> float:
    """Estimate volume of unit hypersphere via Monte Carlo.

    Known answer: V_d = π^{d/2} / Γ(d/2 + 1)
    Method: sample uniformly in [-1, 1]^d, count fraction inside sphere.
    """
    if quasi:
        # Sobol in [0,1) → shift to [-1,1)
        points = 2.0 * trnrand.sobol(n_points, n_dims, seed=42) - 1.0
    else:
        g = trnrand.Generator(seed=42)
        points = trnrand.uniform(n_points, n_dims, low=-1.0, high=1.0, generator=g)

    # Check which points are inside the unit sphere
    r_squared = (points ** 2).sum(dim=1)
    inside = (r_squared <= 1.0).float().mean().item()

    # Volume = fraction_inside * volume_of_cube
    cube_volume = 2.0 ** n_dims
    return inside * cube_volume


def exact_sphere_volume(d: int) -> float:
    """Exact volume of d-dimensional unit sphere."""
    import math
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


def main():
    dims = 5
    exact = exact_sphere_volume(dims)
    print(f"Estimating volume of {dims}-D unit sphere (exact = {exact:.6f})")
    print()
    print(f"{'N':>8}  {'MC':>10}  {'MC err':>10}  {'QMC':>10}  {'QMC err':>10}  {'QMC/MC':>8}")
    print("-" * 68)

    for log_n in range(8, 17):
        n = 1 << log_n
        mc_est = sphere_volume_mc(n, dims, quasi=False)
        qmc_est = sphere_volume_mc(n, dims, quasi=True)
        mc_err = abs(mc_est - exact)
        qmc_err = abs(qmc_est - exact)
        ratio = qmc_err / mc_err if mc_err > 1e-10 else float("inf")
        print(f"{n:>8}  {mc_est:>10.6f}  {mc_err:>10.6f}  {qmc_est:>10.6f}  {qmc_err:>10.6f}  {ratio:>8.3f}")

    print()
    print("QMC/MC < 1 means quasi-random is more accurate.")
    print("For smooth integrands, QMC converges O(1/N) vs O(1/√N) for MC.")


if __name__ == "__main__":
    main()
