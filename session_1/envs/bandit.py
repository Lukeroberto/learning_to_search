import numpy as np
import scipy as sci
import gym

class Bandit(gym.Env):

    def __init__(self, config):
        self.n_arms = config["num_arms"]
        self.probs = config["probs"]
        
        self.best_prob = np.max(self.probs)

        assert len(self.probs) == self.n_arms

        self.cumulative_reward = np.zeros(self.n_arms)
        self.visits = np.zeros(self.n_arms)
        self.regret = [0.]
    
    
    def step(self, action):
        assert action < self.n_arms and action >= 0
        self.visits[action] += 1
        self.regret.append(self.best_prob - self.probs[action])

        if np.random.random() < self.probs[action]:
            self.cumulative_reward[action] += 1
            return 1
        
        return 0
    
    def arm_avgs(self):
        return self.cumulative_reward / (self.visits + 0.000001)

    def reset(self):
        self.cumulative_reward.fill(0)
        self.regret = [0.]
        self.visits.fill(0)
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def get_regret(self):
        return self.regret