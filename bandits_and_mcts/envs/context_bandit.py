import numpy as np
import scipy as sci
import gym

def get_state_probs(n_states, n_arms):
    state_probs = np.zeros((n_states, n_arms))
    for s in range(n_states):
        state_probs[s, :] = np.random.uniform(0.0, 0.4)
        best_arm = np.random.randint(n_arms)
        state_probs[s, best_arm] = 0.9
    
    return state_probs
class ContextualBandit(gym.Env):

    def __init__(self, config):
        self.n_arms = config["num_arms"]
        self.n_states = config["num_states"]
        
        # nS x nArms matrix
        self.probs = config["state_probs"]

        assert self.probs.shape == (self.n_states, self.n_arms) 

        # Best arms in each state
        self.best_probs = np.max(self.probs, axis=1)
        
        self.cumulative_reward = np.zeros((self.n_states, self.n_arms))
        self.visits = np.zeros((self.n_states, self.n_arms))
        self.regret = [0.]
        self.state = np.random.randint(self.n_states)
    
    def step(self, action):
        assert action < self.n_arms and action >= 0
        
        self.visits[self.state, action] += 1
        self.regret.append(self.best_probs[self.state] - self.probs[self.state, action])
        
        if np.random.random() < self.probs[self.state, action]:
            self.cumulative_reward[self.state, action] += 1
            self.state = np.random.randint(self.n_states)
            return 1, self.state

        # next state
        self.state = np.random.randint(self.n_states)
        return 0, self.state

    def reset(self):
        self.cumulative_reward.fill(0)
        self.regret = [0.]
        self.visits.fill(0) 
        
        self.state = np.random.randint(self.n_states)
        return self.state

    def arm_avgs(self, state):
        return self.cumulative_reward[state, :] / (self.visits[state, :] + 0.00001)

    def seed(self, seed):
        np.random.seed(seed)
    
    def get_regret(self):
        return self.regret
