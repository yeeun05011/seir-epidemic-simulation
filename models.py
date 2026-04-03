import numpy as np
import random


class Agent:
    """
    Represents one agent in the Monte Carlo SEIR model.
    State codes:
    1 = Susceptible
    2 = Exposed
    3 = Infected
    4 = Recovered
    """
    def __init__(self, x, y, state, grid_size):
        self.__x = x
        self.__y = y
        self.__state = state
        self.__grid_size = grid_size

    @property
    def state(self):
        return self.__state

    @state.setter
    def state(self, new_state):
        self.__state = new_state

    def get_pos(self):
        return self.__x, self.__y

    def move(self):
        """
        Move to one of the 4 nearest-neighbour sites
        using periodic boundary conditions.
        """
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.__x = (self.__x + dx) % self.__grid_size
        self.__y = (self.__y + dy) % self.__grid_size


class SimulationGrid:
    """
    Stores the lattice occupancy/state.
    """
    def __init__(self, size):
        self.__size = size
        self.__grid = np.zeros((size, size), dtype=int)

    def update_occupancy(self, agents):
        self.__grid.fill(0)
        for agent in agents:
            x, y = agent.get_pos()
            self.__grid[x, y] = agent.state

    def has_infected_neighbor(self, x, y):
        """
        Check the 4 nearest neighbours for an infected agent.
        """
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.__size
            ny = (y + dy) % self.__size
            if self.__grid[nx, ny] == 3:
                return True
        return False