import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

class MetropolisHastings:
    def __init__(self, target_density, proposal_std=1.0, dim=1, initial=None, burn_in=0):
        """
        Parameters:
        - target_density: function R^d → R, unnormalized density (can be log-density if you modify the ratio)
        - proposal_std: standard deviation for Gaussian proposal (scalar or array of shape [dim])
        - dim: number of dimensions
        - initial: starting point (default is origin)
        - burn_in: number of initial samples to discard
        """
        self.target_density = target_density
        self.dim = dim
        self.proposal_std = np.atleast_1d(proposal_std)
        self.burn_in = burn_in
        self.current = np.zeros(dim) if initial is None else np.array(initial, dtype=float)
        self.samples = []

    def step(self):
        """Perform one Metropolis-Hastings step."""
        proposal = self.current + np.random.normal(0, self.proposal_std, size=self.dim)
        p_current = self.target_density(self.current)
        p_proposal = self.target_density(proposal)

        if p_current == 0:
            accept_prob = 1.0
        else:
            accept_prob = min(1.0, p_proposal / p_current)

        if np.random.rand() < accept_prob:
            self.current = proposal
        return self.current

    def run(self, num_samples):
        """Run the algorithm and return samples (after burn-in)."""
        all_samples = []
        for _ in range(num_samples + self.burn_in):
            sample = self.step()
            all_samples.append(sample.copy())
        self.samples = np.array(all_samples[self.burn_in:])
        return self.samples

    def get_samples(self):
        return np.array(self.samples)

    def plot_trajectory(self, n_points=500, show=True):
        """
        Plot the trajectory of the Markov chain in 2D (first two components).

        Parameters:
        - n_points: number of steps to plot (default: 500)
        - show: whether to call plt.show() immediately
        """
        if self.dim > 3:
            raise ValueError("Dimension is too big to plot")

        samples = self.samples[:n_points]

        x = samples[:, 0]
        y = samples[:, 1]

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', markersize=2, linewidth=0.5, alpha=0.7)
        plt.title(f"Metropolis-Hastings Trajectory ({n_points} steps)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.axis("equal")
        if show:
            plt.show()

    def plot_samples(self):
        if self.dim > 3:
            raise ValueError("Dimension is too big to plot")

        sns.kdeplot(x=self.samples[:, 0], y=self.samples[:, 1], fill=True, cmap="mako")
        plt.axis("equal")
        plt.title("Metropolis-Hastings Samples (2D Gaussian)")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()



def target_density(x):
    x = np.asarray(x)
    mu = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    inv_cov = np.linalg.inv(cov)
    diff = x - mu
    exponent = -0.5 * diff @ inv_cov @ diff
    return np.exp(exponent)

mh = MetropolisHastings(target_density, proposal_std=0.5, dim=2, burn_in=1000, initial=[10.1,14.0])
samples = mh.run(5000)


mh.plot_trajectory()

mh.plot_samples()























