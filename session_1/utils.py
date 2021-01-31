import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

def run_simulation(algo, env, n_steps=1000, n_trials=1):
    regrets = np.zeros((n_trials, n_steps))
    for trial in range(n_trials):
        env.reset()
        for step in range(n_steps):
            env.step(algo(env))
        regrets[trial] = np.cumsum(env.get_regret())
    return regrets

def plot_simulation(trials, env):
    fig, axs = plt.subplots(1, 2)

    axs[0].bar(range(env.n_arms), env.visits / env.visits.sum()) 
    axs[0].set_title("Arm averages")

    num_steps = np.arange(trials.shape[1])
    mean = np.mean(trials, axis=0)
    # std = np.std(trials, axis=0)
    axs[1].plot(num_steps, mean)
    axs[1].plot(trials.T, color="blue", alpha=0.1)
    axs[1].set_title("Cumulative Regret")
    
    return axs[1]
    


            