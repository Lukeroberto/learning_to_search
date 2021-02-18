import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import seaborn as sns

def color(x):
    return plt.get_cmap("Set1")(x)

def create_gif(loc, n):
    import imageio
    images = []
    for i in range(1, n + 1):
        images.append(imageio.imread(loc + f"{i}.png"))
    imageio.mimsave('assets/solution.gif', images)
    
def run_simulation(algo, env, n_steps=1000, n_trials=1):
    regrets = np.zeros((n_trials, n_steps + 1))
    
    if (len(env.probs.shape) > 1):
        avgs = np.zeros((n_trials, env.n_states, env.n_arms))
    else: 
        avgs = np.zeros((n_trials, env.n_arms))
    for trial in range(n_trials):
        env.reset()
        for step in range(n_steps):
            env.step(algo(env))
        regrets[trial] = np.cumsum(env.get_regret())
        avgs[trial] = env.visits / env.visits.sum()
    return avgs, regrets

def plot_avg_reward(axs, env):
    axs.bar(range(env.n_arms), env.cumulative_reward / (env.visits + 0.001))
    axs.set_title("Average Reward")
    axs.set_xlabel("Arm")
    axs.set_xticks(range(env.n_arms))
    axs.set_ylim([0, 1])

def plot_cumulative_reward(axs, cumulative_reward):
    axs.plot(cumulative_reward)
    axs.set_title("Cumulative Reward")
    axs.set_xlabel("Step")
    axs.set_ylim([0, len(cumulative_reward)])

def plot_cumulative_regret(axs, avg_reward, env):
    cumulative_r = np.cumsum(env.regret)
    steps = np.arange(len(cumulative_r))
    axs.plot(steps, cumulative_r, label="You")
    axs.plot(steps, avg_reward * steps, label="Random")
    axs.set_title("Regret over time")

def plot_simulation(trials, avgs, env):
    names = ["Random", "e-Greedy", "UCB"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Compare arm pulls
    axs[0].bar(range(env.n_arms), avgs.mean(axis=0), color=color(1)) 
    axs[0].set_title("Arm Pull Averages")
    axs[0].set_xlabel("Arm")
    axs[0].set_xticks(range(env.n_arms))
    axs[0].set_ylim([0, 1])

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

def plot_contextual_simulation(trials, avgs, env):
    names = ["Random", "e-Greedy", "UCB"]
    fig, axs = plt.subplots(1, 2, figsize=(15,8))

    # State, action visitation
    action_norm = env.visits.sum(axis=1, keepdims=True)
    sns.heatmap(env.visits / action_norm, cmap="Blues", ax=axs[0], vmax=1, vmin=0, cbar=False, annot=True)
    axs[0].set_title("Arm averages")
    axs[0].set_ylabel("State/Context")
    axs[0].set_xlabel("Arm")
    # axs[0].set_xticks(range(env.n_arms))

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
    


            