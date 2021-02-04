import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import seaborn as sns

def color(x):
    return plt.get_cmap("Set1")(x)

def run_simulation(algo, env, n_steps=1000, n_trials=1):
    regrets = np.zeros((n_trials, n_steps + 1))
    for trial in range(n_trials):
        env.reset()
        for step in range(n_steps):
            env.step(algo(env))
        regrets[trial] = np.cumsum(env.get_regret())
    return regrets

def plot_simulation(trials, env):
    names = ["Random", "e-Greedy", "UCB"]
    fig, axs = plt.subplots(1, 2)

    # Compare arm pulls
    axs[0].bar(range(env.n_arms), env.visits / env.visits.sum(), color=color(1)) 
    axs[0].set_title("Arm averages")

    # Compare regrets
    for i, trial in enumerate(trials):
        num_steps = np.arange(trial.shape[1])
        mean = np.mean(trial, axis=0)
        # std = np.std(trials, axis=0)
        axs[1].plot(num_steps, mean, color=color(2 + i), label=names[i])
        axs[1].plot(trial.T, color=color(2 + i), alpha=0.01)
        axs[1].set_title("Cumulative Regret")
        axs[1].legend()

    fig.tight_layout()

def plot_contextual_simulation(trials, env):
    names = ["Random", "e-Greedy", "UCB"]
    fig, axs = plt.subplots(1, 2)

    # State, action visitation
    sns.heatmap(env.visits / env.visits.sum(axis=0), cmap="Blues", ax=axs[0], vmax=1, vmin=0, cbar=False, annot=True)
    axs[0].set_title("Arm averages")

    # Compare regrets
    for i, trial in enumerate(trials):
        num_steps = np.arange(trial.shape[1])
        mean = np.mean(trial, axis=0)
        # std = np.std(trials, axis=0)
        axs[1].plot(num_steps, mean, color=color(2 + i), label=names[i])
        axs[1].plot(trial.T, color=color(2 + i), alpha=0.05)
        axs[1].set_title("Cumulative Regret")
        axs[1].legend()

    fig.tight_layout()
    


            