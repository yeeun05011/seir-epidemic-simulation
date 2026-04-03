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
        if new_state not in [1, 2, 3, 4]:
            raise ValueError("Invalid state.")
        self.__state = new_state

    def get_pos(self):
        return self.__x, self.__y

    def move(self, grid):
        """
        Move to one of the 4 nearest-neighbour sites
        using periodic boundary conditions.
        """
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        new_x = (self.__x + dx) % self.__grid_size
        new_y = (self.__y + dy) % self.__grid_size

        if grid.is_empty(new_x, new_y):
            grid.set_cell(self.__x, self.__y, 0)
            self.__x, self.__y = new_x, new_y
            grid.set_cell(self.__x, self.__y, self.__state)


class SimulationGrid:
    """
    Stores the lattice occupancy/state.
    """
    def __init__(self, size):
        self.__size = size
        self.__grid = np.zeros((size, size), dtype=int)

    def is_empty(self, x, y):
        return self.__grid[x, y] == 0

    def set_cell(self, x, y, value):
        self.__grid[x, y] = value

    def update_occupancy(self, agents):
        """
        Optional full refresh of the lattice from the agent list.
        """
        self.__grid.fill(0)
        for agent in agents:
            x, y = agent.get_pos()
            self.__grid[x, y] = agent.state

    def has_infected_neighbor(self, x, y):
        """
        Check whether at least one nearest neighbour is infected.
        """
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.__size
            ny = (y + dy) % self.__size
            if self.__grid[nx, ny] == 3:
                return True
        return False

    def count_infected_neighbors(self, x, y):
        """
        Count the number of infected nearest neighbours.
        """
        count = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.__size
            ny = (y + dy) % self.__size
            if self.__grid[nx, ny] == 3:
                count += 1
        return count

    def get_empty_position(self):
        """
        Return a random empty lattice site.
        """
        while True:
            x = random.randint(0, self.__size - 1)
            y = random.randint(0, self.__size - 1)
            if self.is_empty(x, y):
                return x, y