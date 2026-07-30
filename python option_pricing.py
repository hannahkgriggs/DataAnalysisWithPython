import numpy as np
from scipy.stats import norm

def black_scholes_european_call(S, K, T, r, sigma):
    """
    Calculates exact European Call option price using the Black-Scholes formula.
    
    Parameters:
    S     : Current asset/stock price (Spot price)
    K     : Strike price
    T     : Time to maturity in years
    r     : Risk-free annual interest rate (e.g., 0.05 for 5%)
    sigma : Volatility of the underlying asset (e.g., 0.20 for 20%)
    """
    # Step 1: Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Step 2: Cumulative Normal Distribution CDF N(d1) and N(d2)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def monte_carlo_european_call(S, K, T, r, sigma, num_simulations=100_000, seed=42):
    """
    Estimates European Call option price via Monte Carlo Simulation (Geometric Brownian Motion).
    Includes Antithetic Variates for Variance Reduction.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Step 1: Generate standard normal random variables Z ~ N(0, 1)
    # Using antithetic variates: generate N/2 random numbers and pair them with -Z
    n_half = num_simulations // 2
    Z = np.random.standard_normal(n_half)
    Z_antithetic = np.concatenate([Z, -Z])
    
    # Step 2: Simulate terminal stock price S_T under Risk-Neutral Measure:
    # S_T = S * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z_antithetic
    S_T = S * np.exp(drift + diffusion)
    
    # Step 3: Calculate payoff at maturity: max(S_T - K, 0)
    payoffs = np.maximum(S_T - K, 0)
    
    # Step 4: Discount average payoff back to present value: PV = e^(-rT) * E[Payoff]
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    
    # Step 5: Calculate Standard Error of the Monte Carlo estimate
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(num_simulations)
    
    return mc_price, std_err


# --- DEMO & VERIFICATION SCRIPT ---
if __name__ == "__main__":
    # Input Parameters
    Spot = 100.0    # Stock price today = $100
    Strike = 100.0  # Strike price = $100 (At-the-money)
    Time = 1.0      # 1 year to expiration
    Rate = 0.05     # 5% risk-free rate
    Vol = 0.20      # 20% annual volatility
    
    # 1. Exact Analytical Price (Black-Scholes)
    bs_price = black_scholes_european_call(Spot, Strike, Time, Rate, Vol)
    
    # 2. Numerical Simulation Price (Monte Carlo)
    mc_price, std_err = monte_carlo_european_call(Spot, Strike, Time, Rate, Vol, num_simulations=500_000)
    
    # Output Results
    print("=" * 55)
    print("      QUANTITATIVE OPTION PRICING VERIFICATION       ")
    print("=" * 55)
    print(f"Exact Black-Scholes Price : ${bs_price:.4f}")
    print(f"Monte Carlo Simulated Price: ${mc_price:.4f}")
    print(f"Standard Error (MC)        : ±${std_err:.4f}")
    print(f"Absolute Pricing Error     : ${abs(bs_price - mc_price):.4f}")
    print("=" * 55)
