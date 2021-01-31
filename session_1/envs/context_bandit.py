import numpy as np
import scipy as sci
import gym

class ContextualBandit(gym.Env):

    def __init__(self, config):
        self.n_arms = config["num_arms"]
        self.n_states = config["num_states"]
        
        # nS x nArms matrix
        self.probs = config["stat_probs"]

        assert self.probs.shape == (self.n_arms, self.n_states) 

        # Best arms in each state
        self.best_probs = np.max(self.probs, axis=1)
    
    def step(self, action):
        return 0, 0