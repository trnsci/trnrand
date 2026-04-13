"""
Gamma-distributed waiting times — Erlang / reliability interpretation.

If a system needs k independent exponentially-distributed events to
complete (rate λ each), the total time is Erlang(k, λ), which equals
Gamma(shape=k, scale=1/λ). This example verifies that directly by
comparing:

  • direct samples from trnrand.gamma(shape=k, scale=1/λ)
  • sum of k independent Exponential(λ) draws

The two distributions should match in mean, variance, and shape.

Applications: time until the k-th component failure (reliability),
k-stage service time (queuing), k-event burst detection (neural spiking).

Usage:
    python examples/gamma_waiting_times.py
"""

import torch
import trnrand


def direct_gamma(shape: int, rate: float, n: int, seed: int) -> torch.Tensor:
    g = trnrand.Generator(seed=seed)
    return trnrand.gamma(n, shape=float(shape), scale=1.0 / rate, generator=g)


def sum_of_exponentials(shape: int, rate: float, n: int, seed: int) -> torch.Tensor:
    g = trnrand.Generator(seed=seed)
    # Draw shape*n exponentials, sum in groups of `shape`.
    samples = trnrand.exponential(n * shape, rate=rate, generator=g)
    return samples.reshape(n, shape).sum(dim=1)


def main():
    k = 4            # stages / components to complete
    rate = 0.5       # each stage completes at rate 0.5 per unit time → mean 2
    n_samples = 200_000

    direct = direct_gamma(k, rate, n_samples, seed=42).double()
    summed = sum_of_exponentials(k, rate, n_samples, seed=99).double()

    theory_mean = k / rate            # Erlang(k, λ) mean = k/λ
    theory_var = k / (rate ** 2)      #                variance = k/λ²

    print("Gamma / Erlang waiting times — two equivalent formulations")
    print("=" * 60)
    print(f"Shape k = {k}, rate λ = {rate}")
    print(f"Theoretical mean     = {theory_mean:.4f}")
    print(f"Theoretical variance = {theory_var:.4f}")
    print()
    print(f"Direct Gamma(k, 1/λ):       mean = {direct.mean().item():.4f}, "
          f"var = {direct.var().item():.4f}")
    print(f"Sum of k Exponential(λ):    mean = {summed.mean().item():.4f}, "
          f"var = {summed.var().item():.4f}")
    print()
    # Compare empirical quantiles at a few levels.
    quantile_levels = torch.tensor([0.25, 0.5, 0.75, 0.95], dtype=torch.float64)
    q_direct = torch.quantile(direct, quantile_levels).tolist()
    q_summed = torch.quantile(summed, quantile_levels).tolist()
    print("Empirical quantile comparison:")
    print(f"{'level':>8} {'direct':>12} {'sum-of-exp':>14} {'abs diff':>12}")
    for lvl, qd, qs in zip(quantile_levels.tolist(), q_direct, q_summed):
        print(f"{lvl:>8.2f} {qd:>12.4f} {qs:>14.4f} {abs(qd - qs):>12.4f}")
    print()
    print("Agreement across quantiles confirms that Gamma(k, 1/λ) =")
    print("sum of k iid Exponential(λ) — the Erlang identity.")


if __name__ == "__main__":
    main()
