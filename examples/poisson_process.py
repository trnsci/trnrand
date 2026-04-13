"""
Homogeneous Poisson process simulation — two equivalent views.

A rate-λ Poisson process over [0, T] can be simulated two ways:

  1. Count view: the number of arrivals in [0, T] is Poisson(λT).
  2. Event-time view: inter-arrival times are iid Exponential(λ).

These two views are equivalent — the simulated arrival counts and
event-time counts should agree in distribution. This example shows
both paths side-by-side.

Usage:
    python examples/poisson_process.py
"""

import torch
import trnrand


def count_view(rate: float, horizon: float, n_replications: int, seed: int) -> torch.Tensor:
    """Return the arrival count in [0, T] for each replication."""
    g = trnrand.Generator(seed=seed)
    return trnrand.poisson(n_replications, lam=rate * horizon, generator=g)


def event_time_view(
    rate: float, horizon: float, n_replications: int, seed: int, max_events: int = 1000
) -> torch.Tensor:
    """For each replication, count arrivals by accumulating exponential gaps."""
    g = trnrand.Generator(seed=seed)
    counts = torch.zeros(n_replications, dtype=torch.int64)
    for i in range(n_replications):
        inter_arrivals = trnrand.exponential(max_events, rate=rate, generator=g)
        arrival_times = torch.cumsum(inter_arrivals, dim=0)
        counts[i] = int((arrival_times < horizon).sum().item())
    return counts


def main():
    rate = 2.5       # events per unit time
    horizon = 10.0   # observe for 10 units → expect ~25 events per replication
    n_replications = 20_000

    counts_a = count_view(rate, horizon, n_replications, seed=42)
    counts_b = event_time_view(rate, horizon, n_replications=500, seed=42)

    print("Homogeneous Poisson process — count vs event-time equivalence")
    print("=" * 62)
    print(f"Rate λ = {rate}, horizon T = {horizon}, expected Poisson(λT) mean = {rate*horizon}")
    print()
    print(f"Count view   (n={n_replications}):  "
          f"mean = {counts_a.double().mean().item():.3f}, "
          f"var = {counts_a.double().var().item():.3f}")
    print(f"Event-time   (n=500):    "
          f"mean = {counts_b.double().mean().item():.3f}, "
          f"var = {counts_b.double().var().item():.3f}")
    print()
    print(f"Both means should be ≈ {rate*horizon} (Poisson(λT) has mean = variance = λT).")
    print("Agreement between the two views confirms the Exponential-interarrival")
    print("and Poisson-count formulations of the same process.")


if __name__ == "__main__":
    main()
