import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from scipy.stats import norm

# Black-Scholes price for a call option
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Newton-Raphson method to calculate implied volatility
def implied_volatility_newton(S, K, T, r, market_price, tol=1e-6, max_iter=100):
    sigma = 0.5  # Adjusted initial guess
    for i in range(max_iter):
        price = black_scholes_call(S, K, T, r, sigma)
        vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        price_diff = price - market_price

        if abs(price_diff) < tol:
            return sigma

        sigma -= price_diff / vega

        if sigma <= 0:  # Ensure sigma remains positive
            sigma = tol

    return None  # Return None if it fails to converge

# Bisection method to calculate implied volatility
def implied_volatility_bisection(S, K, T, r, market_price, tol=1e-6, max_iter=100):
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        price = black_scholes_call(S, K, T, r, mid)

        if abs(price - market_price) < tol:
            return mid

        if price > market_price:
            high = mid
        else:
            low = mid

    return None  # Return None if it fails to converge

# GUI Setup
def calculate_implied_volatility():
    try:
        S = float(entry_S.get())
        K = float(entry_K.get())
        T = float(entry_T.get())
        r = float(entry_r.get()) / 100  # Convert percentage to decimal
        market_price = float(entry_market_price.get())
        method = method_var.get()

        if method == "Newton-Raphson":
            iv = implied_volatility_newton(S, K, T, r, market_price)
        elif method == "Bisection":
            iv = implied_volatility_bisection(S, K, T, r, market_price)
        else:
            raise ValueError("Invalid method selected")

        if iv is not None:
            result_label.config(text=f"Implied Volatility: {iv:.6f}", fg="green")
        else:
            result_label.config(text="Calculation did not converge. Try different inputs.", fg="red")

    except ValueError as e:
        result_label.config(text=str(e), fg="red")
    except Exception:
        result_label.config(text="Invalid input. Please check your entries.", fg="red")

root = tk.Tk()
root.title("Implied Volatility Calculator")
root.geometry("500x500")
root.resizable(False, False)

# Header
header = tk.Label(root, text="Implied Volatility Calculator", font=("Arial", 16, "bold"), fg="blue")
header.pack(pady=10)

# Input fields
frame_inputs = tk.Frame(root)
frame_inputs.pack(pady=10)

label_S = tk.Label(frame_inputs, text="Spot Price (S):", font=("Arial", 12))
label_S.grid(row=0, column=0, padx=5, pady=5, sticky="e")
entry_S = tk.Entry(frame_inputs, font=("Arial", 12))
entry_S.grid(row=0, column=1, padx=5, pady=5)

label_K = tk.Label(frame_inputs, text="Strike Price (K):", font=("Arial", 12))
label_K.grid(row=1, column=0, padx=5, pady=5, sticky="e")
entry_K = tk.Entry(frame_inputs, font=("Arial", 12))
entry_K.grid(row=1, column=1, padx=5, pady=5)

label_T = tk.Label(frame_inputs, text="Time to Maturity (T in years):", font=("Arial", 12))
label_T.grid(row=2, column=0, padx=5, pady=5, sticky="e")
entry_T = tk.Entry(frame_inputs, font=("Arial", 12))
entry_T.grid(row=2, column=1, padx=5, pady=5)

label_r = tk.Label(frame_inputs, text="Risk-Free Rate (r in %):", font=("Arial", 12))
label_r.grid(row=3, column=0, padx=5, pady=5, sticky="e")
entry_r = tk.Entry(frame_inputs, font=("Arial", 12))
entry_r.grid(row=3, column=1, padx=5, pady=5)

label_market_price = tk.Label(frame_inputs, text="Market Price of Option:", font=("Arial", 12))
label_market_price.grid(row=4, column=0, padx=5, pady=5, sticky="e")
entry_market_price = tk.Entry(frame_inputs, font=("Arial", 12))
entry_market_price.grid(row=4, column=1, padx=5, pady=5)

# Method selection
label_method = tk.Label(root, text="Select Method:", font=("Arial", 12))
label_method.pack(pady=5)
method_var = tk.StringVar(value="Newton-Raphson")
method_dropdown = ttk.Combobox(root, textvariable=method_var, values=["Newton-Raphson", "Bisection"], state="readonly", font=("Arial", 12))
method_dropdown.pack(pady=5)

# Calculate button
calculate_button = tk.Button(root, text="Calculate Implied Volatility", font=("Arial", 12), bg="blue", fg="white", command=calculate_implied_volatility)
calculate_button.pack(pady=20)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Footer
footer = tk.Label(root, text="Powered by Python", font=("Arial", 10), fg="gray")
footer.pack(side="bottom", pady=10)

root.mainloop()
