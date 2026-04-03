import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys

def seir_equations(y, t, beta, sigma, gamma):
    """
    Defines the differential equations for the SEIR model.
    s: susceptible, e: exposed, i: infected, r: recovered
    """
    s, e, i, r = y
    
    # SEIR coupled equations
    dsdt = -beta * i * s
    dedt = beta * i * s - sigma * e
    didt = sigma * e - gamma * i
    drdt = gamma * i
    
    return [dsdt, dedt, didt, drdt]

def calculate_r0(beta, gamma, s0 = 1.0):
    if gamma <= 0:
        raise ValueError("gamma must me positive.")
    return (beta /gamma) * s0

def solve_seir_case(beta, sigma, gamma, initial_state, time_points):
    results = odeint(seir_equations, initial_state, time_points, args=(beta, sigma, gamma))

    totals = np.sum(results, axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6):
        raise RuntimeError("Population is not conserved in the ODE solution.")

    return results

def run_part1_experiments():
    print("Part 1: Exploring different R0 regimes")
    
    # Initial conditions (Fraction of population)
    # Start with 99% susceptible and 1% exposed
    s0, e0, i0, r0 = 0.99, 0.01, 0.0, 0.0
    initial_state = [s0, e0, i0, r0]
    
    # Time grid (0 to 100 days)
    # Using 1000 points to ensure smooth curves and consistent types (Feedback 11)
    time_points = np.linspace(0.0, 100.0, 1000)

    cases = [
        {"label": "Disease dies out", "beta": 0.05, "sigma": 1.0, "gamma": 0.1},
        {"label": "Moderate outbreak", "beta": 0.2, "sigma": 1.0, "gamma": 0.1},
        {"label": "Large outbreak", "beta": 1.0, "sigma": 1.0, "gamma": 0.1},
    ]

    plt.figure(figsize=(10,6))


    for case in cases:
        beta = case["beta"]
        sigma = case["sigma"]
        gamma = case["gamma"]
        r0_value = calculate_r0(beta, gamma, s0)

        results = solve_seir_case(beta, sigma, gamma, initial_state, time_points)

        plt.plot(
            time_points,
            results[:, 2],
            label=f'{case["label"]} (R0={r0_value:.2f})'
        )

    plt.title("Comparison of infected population for different R0 values")
    plt.xlabel("Time (days)")
    plt.ylabel("Infected Fraction")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_part1_experiments()