import numpy as np
from scipy.stats import norm


def black_scholes_call(s, k, t, r, sigma):
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)


def monte_carlo_call(s, k, t, r, sigma, num_sims=500_000, seed=42):
    if seed is not None:
        np.random.seed(seed)

    # Antithetic sampling
    z = np.random.standard_normal(num_sims // 2)
    z = np.concatenate([z, -z])

    # Geometric Brownian Motion terminal price
    drift = (r - 0.5 * sigma**2) * t
    diffusion = sigma * np.sqrt(t) * z
    s_t = s * np.exp(drift + diffusion)

    payoffs = np.maximum(s_t - k, 0)
    price = np.exp(-r * t) * np.mean(payoffs)
    std_err = np.exp(-r * t) * np.std(payoffs) / np.sqrt(num_sims)

    return price, std_err


if __name__ == "__main__":
    spot = 100.0
    strike = 100.0
    t = 1.0
    r = 0.05
    vol = 0.20

    bs_price = black_scholes_call(spot, strike, t, r, vol)
    mc_price, std_err = monte_carlo_call(spot, strike, t, r, vol)

    print(f"Black-Scholes Price: ${bs_price:.4f}")
    print(f"Monte Carlo Price:   ${mc_price:.4f} (+/- ${std_err:.4f})")
    print(f"Absolute Diff:       ${abs(bs_price - mc_price):.4f}")
