# Option Price for European Call and Put Options
# This program calculates the price of European call and put options using the Black-Scholes formula.
import math
import numpy as np
from scipy.stats import norm
import pandas as pd
import time
import matplotlib.pyplot as plt
import scipy.stats as stats


# Function to calculate the price of European call and put options using the Black-Scholes formula
# Option Params
S = 100  # Current stock price
K = 100  # Option strike price
T = 1  # Time to expiration in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying stock
q = 0.02  # Dividend yield


# Function to calculate the price of European call and put options using the Black-Scholes formula
def black_scholes(S, K, T, r, sigma, q):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(
        d2
    )
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(
        -d1
    )
    return call_price, put_price


# Calculate call and put prices
call_price_benchmark, put_price_benchmark = black_scholes(S, K, T, r, sigma, q)
# Print the results
print(f"Call Price Benchmark: {call_price_benchmark:.2f}")
print(f"Put Price Benchmark: {put_price_benchmark:.2f}")


# Function to calculate the price of European call and put options using binomial tree
def binomial_tree(S, K, T, r, sigma, q, put=False, american=False, n=100):
    """
    Calculate the price of European call and put options using the binomial tree method.
    Parameters:
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    q : float : Dividend yield
    n : int : Number of time steps
    type : str : 'call' for call option, 'put' for put option
    american : bool : True for American option, False for European option
    Returns:
    float : Option price
    """
    # Calculate parameters for the binomial tree
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    asset_prices = np.zeros((n + 2) * (n + 1) // 2)

    def binomial_index(j, i):
        return (j * (j + 1)) // 2 + i

    for i in range(n + 1):
        asset_prices[binomial_index(n, i)] = S * (u ** (n - i)) * (d**i)

    # Initialize option values at maturity
    option_values = np.maximum(0, asset_prices - K if not put else K - asset_prices)

    # Backward induction to calculate option price
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            if american:
                option_values[binomial_index(j, i)] = max(
                    option_values[binomial_index(j, i)],
                    (
                        p * option_values[binomial_index(j + 1, i)]
                        + (1 - p) * option_values[binomial_index(j + 1, i + 1)]
                    )
                    * math.exp(-r * dt),
                )
            else:
                option_values[binomial_index(j, i)] = (
                    p * option_values[binomial_index(j + 1, i)]
                    + (1 - p) * option_values[binomial_index(j + 1, i + 1)]
                ) * math.exp(-r * dt)

    return option_values[0]


# Calculate call and put prices using binomial tree
n = 100  # Number of time steps
call_price_binomial = binomial_tree(S, K, T, r, sigma, q, american=False, n=100)
# put_price_binomial = binomial_tree(S, K, T, r, sigma, q, put=True, american=False, n=1000)
# Print the results
print(f"Call Price Binomial: {call_price_binomial:.2f}")
# print(f"Put Price Binomial: {put_price_binomial:.2f}")

# Function to calculate the price of call and put options using trinomial tree


def trinomial_tree(
    S, K, T, r, sigma, q, put=False, american=False, n=100, l=math.sqrt(2)
):
    """
    Calculate the price of European call and put options using the trinomial tree method.
    Parameters:
    S : float : Current stock price
    K : float : Option strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    q : float : Dividend yield
    n : int : Number of time steps
    type : str : 'call' for call option, 'put' for put option
    american : bool : True for American option, False for European option
    Returns:
    float : Option price
    """
    # Calculate parameters for the trinomial tree
    dt = T / n
    u = math.exp(l * sigma * math.sqrt(dt))
    d = 1 / u
    m = 1
    p_u = (
        (math.exp((r - q) * (dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))
        / (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))
    ) ** 2

    p_d = (
        (math.exp(sigma * math.sqrt(dt / 2)) - math.exp((r - q) * (dt / 2)))
        / (math.exp(sigma * math.sqrt(dt / 2)) - math.exp(-sigma * math.sqrt(dt / 2)))
    ) ** 2
    p_m = 1 - p_u - p_d

    # print(f"u: {u}, d: {d}, m: {m}")
    # print(f"p_u: {p_u}, p_d: {p_d}, p_m: {p_m}")
    # print(0.5 + (r - q) * math.sqrt(dt) / (2 * sigma))

    # Initialize asset prices at maturity
    asset_prices = np.zeros((n + 1) ** 2)

    def trinomial_index(j, i):
        return j**2 + i

    for i in range(2 * n + 1):
        asset_prices[trinomial_index(n, i)] = S * (u ** (n - i))

    # Initialize option values at maturity
    option_values = np.maximum(0, asset_prices - K if not put else K - asset_prices)

    # Backward induction to calculate option price
    for j in range(n - 1, -1, -1):
        for i in range(2 * j + 1):
            if american:
                option_values[trinomial_index(j, i)] = max(
                    option_values[trinomial_index(j, i)],
                    (
                        p_u * option_values[trinomial_index(j + 1, i)]
                        + p_m * option_values[trinomial_index(j + 1, i + 1)]
                        + p_d * option_values[trinomial_index(j + 1, i + 2)]
                    )
                    * math.exp(-r * dt),
                )
            else:
                option_values[trinomial_index(j, i)] = (
                    p_u * option_values[trinomial_index(j + 1, i)]
                    + p_m * option_values[trinomial_index(j + 1, i + 1)]
                    + p_d * option_values[trinomial_index(j + 1, i + 2)]
                ) * math.exp(-r * dt)

    return option_values[0]


# Calculate call and put prices using trinomial tree
call_price_trinomial = trinomial_tree(S, K, T, r, sigma, q, american=False, l=math.sqrt(2), n=100)
# put_price_trinomial = trinomial_tree(S, K, T, r, sigma, q, put=True, american=False, l=1, n=1000)
# Print the results
print(f"Call Price Trinomial: {call_price_trinomial:.2f}")
# print(f"Put Price Trinomial: {put_price_trinomial:.2f}")


# Plot convergence of binomial and trinomial tree methods
n_values = range(2, 301, 3)

call_prices_binomial = []
put_prices_binomial = []
call_prices_trinomial = []
put_prices_trinomial = []

for n in n_values:
    call_price_binomial = binomial_tree(S, K, T, r, sigma, q, n=n)
    put_price_binomial = binomial_tree(S, K, T, r, sigma, q, put=True, n=n)
    call_price_trinomial = trinomial_tree(S, K, T, r, sigma, q, n=n)
    put_price_trinomial = trinomial_tree(S, K, T, r, sigma, q, put=True, n=n)
    call_prices_binomial.append(call_price_binomial)
    put_prices_binomial.append(put_price_binomial)
    call_prices_trinomial.append(call_price_trinomial)
    put_prices_trinomial.append(put_price_trinomial)

plt.plot(n_values, call_prices_binomial, label="Call Price (Binomial)")
plt.plot(n_values, put_prices_binomial, label="Put Price (Binomial)")
plt.plot(n_values, call_prices_trinomial, label="Call Price (Trinomial)", linestyle="--")
plt.plot(n_values, put_prices_trinomial, label="Put Price (Trinomial)", linestyle="--")
plt.axhline(
    y=call_price_benchmark, color="r", linestyle="--", label="Benchmark Call Price"
)
plt.axhline(
    y=put_price_benchmark, color="g", linestyle="--", label="Benchmark Put Price"
)
plt.xlabel("Number of Time Steps (n)")
plt.ylabel("Option Price")
plt.title("Convergence of Binomial and Trinomial Tree Methods")
plt.legend()
plt.grid()
plt.show()

# Find the number of steps for both models when the error < 0.1%
tolerance = 0.001  # 1% tolerance
n_binomial = None
n_trinomial = None

for i in range(2, len(n_values)):  # Increase the range if necessary
    
    error_binomial = abs(call_prices_binomial[i] - call_price_benchmark) / call_price_benchmark
    error_trinomial = abs(call_prices_trinomial[i] - call_price_benchmark) / call_price_benchmark
    
    if n_binomial is None and error_binomial < tolerance:
        n_binomial = n_values[i]
    if n_trinomial is None and error_trinomial < tolerance:
        n_trinomial = n_values[i]
    
    if n_binomial is not None and n_trinomial is not None:
        break

print(f"Number of steps for Binomial Tree to achieve < 0.1% error: {n_binomial}")
print(f"Number of steps for Trinomial Tree to achieve < 0.1% error: {n_trinomial}")

# Timing for Binomial Tree
start_time_binomial = time.time()
binomial_tree(S, K, T, r, sigma, q, n=n_binomial)
end_time_binomial = time.time()
time_binomial = end_time_binomial - start_time_binomial

# Timing for Trinomial Tree
start_time_trinomial = time.time()
trinomial_tree(S, K, T, r, sigma, q, n=n_trinomial)
end_time_trinomial = time.time()
time_trinomial = end_time_trinomial - start_time_trinomial

print(f"Time taken for Binomial Tree with {n_binomial} steps: {time_binomial:.6f} seconds")
print(f"Time taken for Trinomial Tree with {n_trinomial} steps: {time_trinomial:.6f} seconds")