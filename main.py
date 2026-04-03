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

def run_part1_simulation():
    print("Part 1: Solving SEIR Model (ODE)")
    
    # Input validation and customisation (Feedback 6 & 7)
    try:
        # Default values: beta=1.0, sigma=1.0, gamma=0.1
        beta_input = input("Enter infection rate (beta) [default 1.0]: ") or "1.0"
        sigma_input = input("Enter incubation rate (sigma) [default 1.0]: ") or "1.0"
        gamma_input = input("Enter recovery rate (gamma) [default 0.1]: ") or "0.1"
        
        beta = float(beta_input)
        sigma = float(sigma_input)
        gamma = float(gamma_input)
        
        # Check for invalid negative values
        if beta < 0 or sigma < 0 or gamma < 0:
            raise ValueError("Parameters cannot be negative.")
            
    except ValueError as e:
        print(f"Runtime Error: {e}")
        sys.exit(1)

    # Initial conditions (Fraction of population)
    # Start with 99% susceptible and 1% exposed
    s0, e0, i0, r0 = 0.99, 0.01, 0.0, 0.0
    initial_state = [s0, e0, i0, r0]
    
    # Time grid (0 to 100 days)
    # Using 1000 points to ensure smooth curves and consistent types (Feedback 11)
    time_points = np.linspace(0.0, 100.0, 1000)

    # Solve the system of ODEs
    results = odeint(seir_equations, initial_state, time_points, args=(beta, sigma, gamma))
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, results[:, 0], 'b-', label='Susceptible')
    plt.plot(time_points, results[:, 1], 'y-', label='Exposed')
    plt.plot(time_points, results[:, 2], 'g-', label='Infected')
    plt.plot(time_points, results[:, 3], 'r-', label='Recovered')
    
    r0 = calculate_r0(beta, gamma, s0)
    plt.title(f'SEIR Model Dynamics (R0 = {r0:.2f})')
    plt.xlabel('Time (days)')
    plt.ylabel('Fraction of Population')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_part1_simulation()