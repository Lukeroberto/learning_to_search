import numpy as np
import scipy as sci

def compute_deviation_inequality(samples, true_mean, t):
    n = np.size(samples)

    ave_deviation = (samples - true_mean).sum() / n
    return ave_deviation >= t

def prob_inequality(sampler, n, t):
    valid = np.zeros(100);
    for i in range(100):
        samples, true_mean = sampler(n)
        valid[i] = compute_deviation_inequality(samples, true_mean, t)
    
    return valid.mean()

def sample_trunc_norm(n):
    return sci.stats.truncnorm.rvs(0.1, 5, size=n), sci.stats.truncnorm.mean(0.1, 5)

def hoeffding_estimate(n, t, interval):
    return np.exp(- (2 * n * np.square(t)) / (np.square(interval)))

def run_trial(n, t, interval):
    prob = prob_inequality(sample_trunc_norm, n, t)
    estimate = hoeffding_estimate(n, t, interval)

    return prob, estimate