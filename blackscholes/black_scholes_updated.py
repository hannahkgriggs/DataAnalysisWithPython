import numpy as np
from scipy.stats import norm


class BlackScholesEngine:
    def __init__(self, S, K, T, r, sigma):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self._calculate_d1_d2()

    def _calculate_d1_d2(self):
        if self.T <= 0 or self.sigma <= 0:
            self.d1, self.d2 = 0.0, 0.0
            return

        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def price(self, option_type="call"):
        if option_type.lower() == "call":
            return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif option_type.lower() == "put":
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def greeks(self, option_type="call"):
        pdf_d1 = norm.pdf(self.d1)
        sqrt_T = np.sqrt(self.T)
        is_call = option_type.lower() == "call"

        # shared greeks
        gamma = pdf_d1 / (self.S * self.sigma * sqrt_T)
        vega = self.S * pdf_d1 * sqrt_T

        if is_call:
            delta = norm.cdf(self.d1)
            theta = (- (self.S * pdf_d1 * self.sigma) / (2 * sqrt_T) 
                     - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2))
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        else:
            delta = norm.cdf(self.d1) - 1.0
            theta = (- (self.S * pdf_d1 * self.sigma) / (2 * sqrt_T) 
                     + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2))
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega / 100.0,   # per 1% vol
            "theta": theta / 365.0, # per day
            "rho": rho / 100.0      # per 1% rate
        }

    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type="call", max_iter=100, tol=1e-6):
        # initial guess
        sigma = np.sqrt(2 * np.pi / T) * (market_price / S)

        # newton-raphson solver
        for _ in range(max_iter):
            engine = BlackScholesEngine(S, K, T, r, sigma)
            price = engine.price(option_type)
            vega = engine.greeks(option_type)["vega"] * 100.0

            diff = price - market_price

            if abs(diff) < tol:
                return sigma

            if vega < 1e-8:
                break

            sigma -= diff / vega

        return sigma


if __name__ == "__main__":
    spot = 100.0
    strike = 100.0
    t = 1.0
    r = 0.05
    vol = 0.20

    bs = BlackScholesEngine(spot, strike, t, r, vol)

    print(f"Call Price: ${bs.price('call'):.4f}")
    print(f"Put Price:  ${bs.price('put'):.4f}\n")

    print("Call Greeks:")
    greeks = bs.greeks('call')
    for g, val in greeks.items():
        print(f"  {g:<6}: {val:+.5f}")

    target_price = 10.4506
    calc_iv = BlackScholesEngine.implied_volatility(target_price, spot, strike, t, r, 'call')
    print(f"\nImplied Volatility: {calc_iv * 100:.2f}%")
