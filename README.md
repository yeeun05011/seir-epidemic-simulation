# SEIR Epidemic Simulation Project

This project models the spread of an epidemic using two different approaches:

1. A deterministic SEIR model (ODE-based)
2. A stochastic Monte Carlo simulation

The goal is to compare how these two models behave under similar conditions.



## Features

- SEIR model solved using `odeint`
- Exploration of different R0 values
- Monte Carlo simulation with moving agents
- Comparison between deterministic and stochastic results



## Project Structure

.
├── main.py  
├── models.py  
├── Makefile  
├── requirements.txt  
└── README.md  


## How to Run

Install dependencies:

make install

Run the simulation:

make run



## Model Overview

### SEIR Model

The system is defined by:

ds/dt = -βsi  
de/dt = βsi - σe  
di/dt = σe - γi  
dr/dt = γi  

R0 = β / γ



### Monte Carlo Simulation

- Agents move randomly on a 2D grid
- Each agent has a state (S, E, I, R)
- Infection occurs probabilistically when an infected neighbour is present.
- Transitions happen with probabilities σ and γ
- In the Monte Carlo model, infection is probabilistic and depends on the number of infected neighbours. The exposure probability is given by: P = 1 - (1 - p)^n

## Key Parameters:
- beta: infection rate (ODE model)
- sigma: incubation rate
- gamma: recovery rate
- infection_prob: probability of transmission per contact (Monte Carlo)

## Results

- The ODE model produces smooth curves
- The Monte Carlo simulation shows fluctuations due to randomness
- However, both follow a similar overall trend

## Comparison
The ODE model provides smooth deterministic behaviour,
while the Monte Carlo model introduces stochastic fluctuations
due to random movement and interactions. This leads to differences in peak height and timing between the two models.

## Notes

- A dev branch was used during development
- The main branch contains the final version
- A tag (v2.0-final) marks the final submission version


## Dependencies

- numpy
- scipy
- matplotlib
