# Black-Scholes & Monte Carlo Option Pricing Engine

A high-performance Python engine for pricing European options using both closed-form **Black-Scholes-Merton** equations and risk-neutral **Monte Carlo simulations**. Includes full analytical **Greeks** calculation ($\Delta, \Gamma, \mathcal{V}, \Theta, P$) and an **Implied Volatility solver** using the Newton-Raphson algorithm.

---

## Key Features

- **Analytical Black-Scholes Solutions:** Closed-form valuation for European Calls and Puts using continuous discounting.
- **Analytical Greeks Engine:** Exact calculations for Delta, Gamma, Vega, Theta, and Rho scaled for practical trading metrics (e.g., daily decay, 1% volatility shifts).
- **Implied Volatility Solver:** Fast convergence using the Newton-Raphson root-finding method ($\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)}$).
- **Monte Carlo Simulation:** Simulates terminal asset price paths under Geometric Brownian Motion (GBM).
- **Variance Reduction:** Utilises **Antithetic Variates** ($Z$ and $-Z$) to accelerate simulation convergence and reduce sampling variance.

---

## Mathematical Foundations

### 1. Black-Scholes Analytical Pricing

The exact price of a European call option $C$ and put option $P$ on a non-dividend-paying asset is given by:

$$C = S \cdot N(d_1) - K \cdot e^{-rT} N(d_2)$$

$$P = K \cdot e^{-rT} N(-d_2) - S \cdot N(-d_1)$$

Where the terms $d_1$ and $d_2$ are defined as:

$$d_1 = \frac{\ln(S / K) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

- $S$: Current spot price of the underlying asset
- $K$: Strike price
- $T$: Time to maturity (in years)
- $r$: Risk-free annual interest rate
- $\sigma$: Volatility of the underlying asset
- $N(x)$: Cumulative distribution function (CDF) of the standard normal distribution

---

### 2. The Greeks

The analytical sensitivities of the option price with respect to market variables:

| Greek | Formula | Financial Interpretation |
| :--- | :--- | :--- |
| **Delta ($\Delta$)** | $\frac{\partial C}{\partial S} = N(d_1)$ | Sensitivity of option price to a $\$1$ change in spot price. |
| **Gamma ($\Gamma$)** | $\frac{\partial^2 C}{\partial S^2} = \frac{N'(d_1)}{S \sigma \sqrt{T}}$ | Rate of change of Delta per $\$1$ change in spot price. |
| **Vega ($\mathcal{V}$)** | $\frac{\partial C}{\partial \sigma} = S N'(d_1) \sqrt{T}$ | Sensitivity to a $1\%$ change in volatility. |
| **Theta ($\Theta$)** | $\frac{\partial C}{\partial t}$ | Sensitivity to time decay (expressed per calendar day). |
| **Rho ($P$)** | $\frac{\partial C}{\partial r} = K T e^{-rT} N(d_2)$ | Sensitivity to a $1\%$ change in risk-free rate. |

---

### 3. Geometric Brownian Motion (Monte Carlo)

Under the risk-neutral measure, the terminal stock price $S_T$ at expiration $T$ evolves according to:

$$S_T = S_0 \cdot \exp\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z\right)$$

Where $Z \sim \mathcal{N}(0, 1)$ is a standard normal random variable. 

The present value of the expected payoff is calculated as:

$$C_{\text{MC}} = e^{-rT} \mathbb{E}\left[\max(S_T - K, 0)\right]$$

---

### 4. Implied Volatility (Newton-Raphson Method)

To calculate implied volatility $\sigma_{\text{IV}}$ given an observed market price $P_{\text{market}}$, we iteratively solve for the root of $f(\sigma) = C_{\text{BS}}(\sigma) - P_{\text{market}} = 0$:

$$\sigma_{n+1} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - P_{\text{market}}}{\mathcal{V}(\sigma_n)}$$

Iterating until $|C_{\text{BS}}(\sigma_n) - P_{\text{market}}| < \epsilon$.

---

## Setup & Execution

### Prerequisites
- Python 3.8+
- NumPy
- SciPy

```bash
pip install numpy scipy
