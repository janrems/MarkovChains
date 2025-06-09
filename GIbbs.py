import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GibbsSampler:
    def __init__(self, conditional_samplers, dim=2, initial=None, burn_in=0):
        """
        Parameters:
        - conditional_samplers: list of functions, each taking (x, idx) and returning a sample
          from p(x_i | x_{-i}). One function per coordinate.
        - dim: dimension of the state space
        - initial: starting point (default: origin)
        - burn_in: number of initial samples to discard
        """
        self.conditional_samplers = conditional_samplers
        self.dim = dim
        self.burn_in = burn_in
        self.current = np.zeros(dim) if initial is None else np.array(initial, dtype=float)
        self.samples = []

    def step(self):
        """Perform one full Gibbs update (cycle over all dimensions)."""
        for i in range(self.dim):
            self.current[i] = self.conditional_samplers[i](self.current.copy(), i)
        return self.current

    def run(self, num_samples):
        """Run the Gibbs sampler and return samples after burn-in."""
        all_samples = []
        for _ in range(num_samples + self.burn_in):
            sample = self.step()
            all_samples.append(sample.copy())
        self.samples = np.array(all_samples[self.burn_in:])
        return self.samples

    def get_samples(self):
        return np.array(self.samples)

    def plot_trajectory(self, n_points=500, show=True):

        if self.dim > 3:
            raise ValueError("Dimension is too big to plot")

        samples = self.samples[:n_points]
        x = samples[:, 0]
        y = samples[:, 1]
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', markersize=2, linewidth=0.5, alpha=0.7)
        plt.title(f"Gibbs Sampler Trajectory ({n_points} steps)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.axis("equal")
        if show:
            plt.show()

    def plot_samples(self):
        if self.dim > 2:
            raise ValueError("Dimension is too big to plot")

        sns.kdeplot(x=self.samples[:, 0], y=self.samples[:, 1], fill=True, cmap="crest")
        plt.axis("equal")
        plt.title("Gibbs Sampler Samples")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()



# Example: Bivariate normal N(0, Σ), where Σ has correlation rho
rho = 0.9

def conditional_sampler_0(x, _):
    mu = rho * x[1]
    return np.random.normal(mu, np.sqrt(1 - rho**2))

def conditional_sampler_1(x, _):
    mu = rho * x[0]
    return np.random.normal(mu, np.sqrt(1 - rho**2))

gibbs = GibbsSampler(
    conditional_samplers=[conditional_sampler_0, conditional_sampler_1],
    dim=2,
    burn_in=0, initial=[10,14]
)

samples = gibbs.run(5000)
gibbs.plot_trajectory()
gibbs.plot_samples()












