import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

from models import Agent, SimulationGrid


def seir_equations(y, t, beta, sigma, gamma):
    """
    Defines the differential equations for the reduced SEIR model.
    s: susceptible, e: exposed, i: infected, r: recovered
    """
    s, e, i, r = y

    dsdt = -beta * i * s
    dedt = beta * i * s - sigma * e
    didt = sigma * e - gamma * i
    drdt = gamma * i

    return [dsdt, dedt, didt, drdt]


def calculate_r0(beta, gamma, s0=1.0):
    """
    Compute the basic reproduction number approximation.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    return (beta / gamma) * s0


def solve_seir_case(beta, sigma, gamma, initial_state, time_points):
    """
    Solve one deterministic SEIR case and validate conservation.
    """
    results = odeint(seir_equations, initial_state, time_points, args=(beta, sigma, gamma))
    
    # Test: population conservation
    totals = np.sum(results, axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6):
        raise RuntimeError("Population is not conserved in the ODE solution.")

    return results


def run_part1_experiments():
    """
    Compare several ODE cases with different R0 values.
    """
    print("Part 1: Exploring different R0 regimes")

    s0, e0, i0, r0 = 0.99, 0.01, 0.0, 0.0
    initial_state = [s0, e0, i0, r0]
    time_points = np.linspace(0.0, 100.0, 1000)

    cases = [
        {"label": "Disease dies out", "beta": 0.05, "sigma": 1.0, "gamma": 0.1},
        {"label": "Moderate outbreak", "beta": 0.2, "sigma": 1.0, "gamma": 0.1},
        {"label": "Large outbreak", "beta": 1.0, "sigma": 1.0, "gamma": 0.1},
    ]

    plt.figure(figsize=(10, 6))

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

def compute_exposure_probability(infected_neighbors, base_prob):
    """
    Probability of becoming exposed given number of infected neighbours.
    Uses: P = 1 - (1 - p)^n
    """
    return 1 - (1 - base_prob) ** infected_neighbors

def run_part2_monte_carlo(
    L=50,
    N=250,
    steps=1000,
    sigma=0.1,
    gamma=0.01,
    infection_prob=0.3,
    initial_exposed_fraction=0.05,
    seed=42
):
    """
    Run the Monte Carlo SEIR simulation on a 2D lattice.
    Infection is now probabilistic rather than guaranteed.
    """
    print("Part 2: Monte Carlo SEIR Model")

    if not (0.0 <= infection_prob <= 1.0):
        raise ValueError("infection_prob must be between 0 and 1.")
    if not (0.0 <= initial_exposed_fraction <= 1.0):
        raise ValueError("initial_exposed_fraction must be between 0 and 1.")
    if sigma < 0.0 or gamma < 0.0:
        raise ValueError("sigma and gamma must be non-negative.")
    if N > L * L:
        raise ValueError("Number of agents exceeds number of lattice sites.")

    random.seed(seed)
    np.random.seed(seed)

    grid = SimulationGrid(L)
    agents = []

    initial_exposed_count = int(N * initial_exposed_fraction)

    # Initial placement without overlap
    for i in range(N):
        x, y = grid.get_empty_position()
        state = 2 if i < initial_exposed_count else 1

        agent = Agent(x, y, state, L)
        agents.append(agent)
        grid.set_cell(x, y, state)

    s_hist, e_hist, i_hist, r_hist = [], [], [], []

    for _ in range(steps):
        for agent in agents:
            x, y = agent.get_pos()
            state = agent.state

            # Susceptible -> Exposed
            if state == 1:
                infected_neighbors = grid.count_infected_neighbors(x, y)
                if infected_neighbors > 0:
                    p_exposure = compute_exposure_probability(
                        infected_neighbors,
                        infection_prob
                    )
                    if random.random() < p_exposure:
                        agent.state = 2

            # Exposed -> Infected
            elif state == 2:
                if random.random() < sigma:
                    agent.state = 3

            # Infected -> Recovered
            elif state == 3:
                if random.random() < gamma:
                    agent.state = 4

            # Immediately update the grid to reflect the new state
            x, y = agent.get_pos()
            grid.set_cell(x, y, agent.state)

            # Move after health update
            agent.move(grid)

        # Count states only after all agents have been updated
        counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for agent in agents:
            counts[agent.state] += 1

        if sum(counts.values()) != N:
            raise RuntimeError("Population is not conserved in Monte Carlo simulation.")

        s_hist.append(counts[1])
        e_hist.append(counts[2])
        i_hist.append(counts[3])
        r_hist.append(counts[4])

    return s_hist, e_hist, i_hist, r_hist


def plot_part2_results(s_hist, e_hist, i_hist, r_hist):
    """
    Plot Monte Carlo compartment populations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(s_hist, label="Susceptible")
    plt.plot(e_hist, label="Exposed")
    plt.plot(i_hist, label="Infected")
    plt.plot(r_hist, label="Recovered")
    plt.title("Monte Carlo SEIR simulation")
    plt.xlabel("Monte Carlo step")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_ode_and_mc():
    """
    Compare deterministic ODE and stochastic Monte Carlo results
    using the infected population.
    """
    print("Comparing ODE vs Monte Carlo")

    beta = 0.2
    sigma = 1.0
    gamma = 0.1

    # ODE setup
    s0, e0, i0, r0 = 0.99, 0.01, 0.0, 0.0
    initial_state = [s0, e0, i0, r0]
    time_points = np.linspace(0.0, 100.0, 1000)

    ode_results = solve_seir_case(beta, sigma, gamma, initial_state, time_points)

    # Monte Carlo setup
    s_hist, e_hist, i_hist, r_hist = run_part2_monte_carlo()

    mc_steps = np.arange(len(i_hist))
    N = 250
    i_hist_normalised = np.array(i_hist) / N

    plt.figure(figsize=(10, 6))
    plt.plot(time_points, ode_results[:, 2], label="ODE Infected", linewidth=2)
    plt.plot(mc_steps, i_hist_normalised, label="MC Infected", alpha=0.7)

    plt.title("ODE vs Monte Carlo Comparison (Infected)")
    plt.xlabel("Time / Steps")
    plt.ylabel("Fraction Infected")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_r0():
    """
    Basic test for R0 calculation.
    """
    expected = 2.0
    actual = calculate_r0(0.2, 0.1)
    if not np.isclose(actual, expected):
        raise RuntimeError("R0 test failed.")


def test_exposure_probability():
    """
    Basic test for probability bounds.
    """
    p = compute_exposure_probability(2, 0.3)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError("Exposure probability out of bounds.")


if __name__ == "__main__":
    test_r0()
    test_exposure_probability()

    run_part1_experiments()

    s_hist, e_hist, i_hist, r_hist = run_part2_monte_carlo()
    plot_part2_results(s_hist, e_hist, i_hist, r_hist)

    compare_ode_and_mc()