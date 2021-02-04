import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def value_iteration(env, gamma=1, tol=0.01):
    v = np.zeros(env.observation_space.n)
    delta = 1
    
    r = lambda s, a: env.P[s][a][0][2]
    T = lambda s, a: env.P[s][a][0][1]
    while delta > tol:
        delta = 0
        for s in range(env.observation_space.n):
            v_temp = np.copy(v)

            # v(s) <- max_a \sum_{s'} P^s_{ss'} [R^a_{ss'} + V(s')]
            v[s] = np.max([r(s, a) + gamma*v[T(s, a)] for a in range(env.action_space.n)])
                    
            delta = max(delta, np.linalg.norm(v - v_temp))
    
    return v

def plot_value_function(v, env):
    sns.heatmap(v.reshape((env.nrow, env.ncol)), cmap="Blues", cbar=False, yticklabels=False, xticklabels=False,annot=True)
    plt.show()