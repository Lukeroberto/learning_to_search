import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

import matplotlib.pyplot as plt 
import seaborn as sns

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if (res[r_new][c_new] != 'H'):
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size, is_slippery=False):
        desc = generate_random_map(size=size)
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            curletter = desc[row, col]
            done = bytes(newletter) in b'GH'
            reward = float(newletter == b'G')
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append((
                                    1. / 3.,
                                    *update_probability_matrix(row, col, b)
                                ))
                        else:
                            li.append((
                                1., *update_probability_matrix(row, col, a)
                            ))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def goal_state(self):
        return self.nS - 1

    def get_action(self, a):
        if a == 0:
            return "LEFT"
        elif a == 1:
            return "DOWN"
        elif a == 2: 
            return "RIGHT"
        elif a == 3:
            return "UP"

    def is_terminal(self, s):
        row, col = self.s // self.ncol, self.s % self.ncol
        return True if self.desc[row, col] in b'GH' else False
        
    def render(self, save=False, ind=0, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        
        lake = np.zeros((self.ncol, self.nrow))
        
        for i, r in enumerate(desc):
            for j, c in enumerate(r):
                if (desc[i][j] == "S"):
                    lake[i, j] = 1
                elif (desc[i][j] == "F"):
                    lake[i, j] = 1
                elif (desc[i][j] == "H"):
                    lake[i, j] = 2
                elif (desc[i][j] == "G"):
                    lake[i, j] = 3

        # Current pos
        lake[row, col] = 0

        # Custom color palette
        start = sns.color_palette("YlGn", 10)[9]
        b = sns.color_palette("Blues", 4)
        ice = b[1]
        hole = b[3]
        end = sns.color_palette("Set2")[5]
        custom = [start, ice, hole, end]

        # Visualize
        sns.heatmap(lake, cmap=custom, cbar=False, xticklabels=False, yticklabels=False, square=True, vmin=0, vmax=3)
        plt.title(f"Frozen Lake {self.ncol}x{self.nrow}")
        
        if save:
            plt.savefig(f"assets/frozen_lake/step_{ind}.png")
        else:
            plt.show()