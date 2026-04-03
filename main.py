import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import sys
import random
from models import Agent, SimulationGrid

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

def run_part2_monte_carlo():
    print("--- Part 2: Monte Carlo SEIR Model ---")

    # Parameters
    L = 50
    N = 250
    steps = 1000
    sigma = 0.1
    gamma = 0.01

    # Create agents
    agents = []
    for i in range(N):
        x = random.randint(0, L - 1)
        y = random.randint(0, L - 1)
        state = 2 if i < int(N * 0.05) else 1
        agents.append(Agent(x, y, state, L))

    grid = SimulationGrid(L)

    s_hist, e_hist, i_hist, r_hist = [], [], [], []

    for _ in range(steps):
        grid.update_occupancy(agents)

        s_count, e_count, i_count, r_count = 0, 0, 0, 0

        for agent in agents:
            state = agent.state

            if state == 1:
                s_count += 1
            elif state == 2:
                e_count += 1
            elif state == 3:
                i_count += 1
            elif state == 4:
                r_count += 1

            x, y = agent.get_pos()

            if state == 1:
                if grid.has_infected_neighbor(x, y):
                    agent.state = 2
            elif state == 2:
                if random.random() < sigma:
                    agent.state = 3
            elif state == 3:
                if random.random() < gamma:
                    agent.state = 4

            agent.move()

        s_hist.append(s_count)
        e_hist.append(e_count)
        i_hist.append(i_count)
        r_hist.append(r_count)

    plt.figure(figsize=(10, 6))
    plt.plot(s_hist, label="Susceptible")
    plt.plot(e_hist, label="Exposed")
    plt.plot(i_hist, label="Infected")
    plt.plot(r_hist, label="Recovered")
    plt.title("Monte Carlo SEIR Model")
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Number of agents")
    plt.legend()
    plt.grid(True)
    plt.show()

    if __name__ == "__main__":
        run_part1_experiments()
        run_part2_monte_carlo()