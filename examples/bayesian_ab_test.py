"""
Bayesian A/B test with a Beta-binomial conjugate prior.

Two variants A and B have unknown true conversion rates. Each is
observed via a Bernoulli trial. With a uniform Beta(1, 1) prior, the
posterior after observing s successes in n trials is Beta(1+s, 1+n-s).
trnrand.beta then lets us draw from both posteriors to estimate
P(B > A) — the probability that variant B actually converts better.

Usage:
    python examples/bayesian_ab_test.py
"""

import torch

import trnrand


def simulate_trials(true_rate: float, n: int, generator: trnrand.Generator) -> int:
    """Run n Bernoulli trials and return the number of successes."""
    draws = trnrand.bernoulli(n, p=true_rate, generator=generator)
    return int(draws.sum().item())


def posterior_probability_b_beats_a(
    a_successes: int,
    a_trials: int,
    b_successes: int,
    b_trials: int,
    n_samples: int = 100_000,
    seed: int = 42,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Draw posterior samples for both variants and estimate P(B > A)."""
    g = trnrand.Generator(seed=seed)
    posterior_a = trnrand.beta(
        n_samples, alpha=1 + a_successes, beta=1 + a_trials - a_successes, generator=g
    )
    posterior_b = trnrand.beta(
        n_samples, alpha=1 + b_successes, beta=1 + b_trials - b_successes, generator=g
    )
    prob_b_better = (posterior_b > posterior_a).float().mean().item()
    return prob_b_better, posterior_a, posterior_b


def main():
    # Ground truth: B is 10% better than A, neither side knows the real rate.
    true_rate_a, true_rate_b = 0.10, 0.11
    n_trials = 5000

    g = trnrand.Generator(seed=7)
    a_successes = simulate_trials(true_rate_a, n_trials, g)
    b_successes = simulate_trials(true_rate_b, n_trials, g)

    prob_b, post_a, post_b = posterior_probability_b_beats_a(
        a_successes, n_trials, b_successes, n_trials
    )

    def credible_interval(samples: torch.Tensor, level: float = 0.95):
        lo = (1 - level) / 2
        hi = 1 - lo
        return torch.quantile(samples, torch.tensor([lo, hi])).tolist()

    ci_a = credible_interval(post_a)
    ci_b = credible_interval(post_b)

    print("Bayesian A/B test — Beta-binomial conjugate posterior")
    print("=" * 56)
    print(f"Variant A: {a_successes} / {n_trials} conversions (true rate {true_rate_a:.3f})")
    print(f"Variant B: {b_successes} / {n_trials} conversions (true rate {true_rate_b:.3f})")
    print()
    print(f"Posterior mean A: {post_a.mean().item():.4f}   95% CI: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"Posterior mean B: {post_b.mean().item():.4f}   95% CI: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print()
    print(f"P(B > A) = {prob_b:.4f}")
    if prob_b > 0.95:
        print("→ Strong evidence B is the better variant.")
    elif prob_b > 0.8:
        print("→ Moderate evidence B is the better variant.")
    else:
        print("→ Not enough evidence to conclude.")


if __name__ == "__main__":
    main()
