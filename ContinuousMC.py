import numpy as np
import matplotlib.pyplot as plt


class AR1Model:
    def __init__(self, phi, initial_mean=0.0, initial_std=1.0, transition_std=1.0):
        """
        phi: autoregressive coefficient (scalar)
        initial_mean: mean of initial distribution
        initial_std: std deviation of initial distribution
        """
        self.phi = phi
        self.initial_mean = initial_mean
        self.initial_std = initial_std
        self.current_state = None
        self.path = None
        self.transition_std=transition_std

    def reset(self, num_chains=1):
        """
        Reset the model with initial values drawn from N(initial_mean, initial_std^2).
        """
        self.current_state = np.random.normal(self.initial_mean, self.initial_std, size=num_chains)
        self.path = [self.current_state.copy()]

    def reset_to_state(self, state, num_chains=1):
        """
        Reset to a specific state.
        """
        self.current_state = np.ones(num_chains) * state
        self.path = [self.current_state.copy()]

    def step(self):
        """
        Perform one AR(1) step.
        """
        noise = np.random.normal(0, self.transition_std, size=self.current_state.shape)
        next_state = self.phi * self.current_state + noise
        self.current_state = next_state
        self.path.append(self.current_state.copy())

    def simulate(self, steps, num_chains=1, reset=True):
        """
        Simulate the AR(1) process.

        Returns:
            numpy array of shape (num_chains, steps+1)
        """
        if reset:
            self.reset(num_chains=num_chains)
        else:
            self.path = list(self.path)

        for _ in range(steps):
            self.step()

        self.path = np.stack(self.path, axis=1)  # shape (num_chains, steps+1)
        return self.path

    def plot_path(self, chain_index=0):
        """
        Plot a single trajectory.
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")
        path = self.path[chain_index]

        plt.figure(figsize=(10, 4))
        plt.plot(path, marker='o', linestyle='-', color='blue')
        plt.xlabel('Time step')
        plt.ylabel('Value')
        plt.title(f'AR(1) Path (Chain {chain_index})')
        plt.grid(True)
        plt.show()

    def plot_empirical_distribution(self, chain_index=0, bins=30):
        """
        Histogram of values in one chain.
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        path = self.path[chain_index]

        plt.figure(figsize=(6, 4))
        plt.hist(path, bins=bins, density=True, color='skyblue')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Empirical Distribution (Chain {chain_index})')
        plt.grid(True)
        plt.show()

    def plot_average_distribution(self, bins=30):
        """
        Histogram over all values from all chains.
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        values = self.path.flatten()

        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=bins, density=True, color='lightgreen')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Empirical Distribution Over All Chains')
        plt.grid(True)
        plt.show()

    def estimate_transition_kernel(self, x0, n_steps=1, num_samples=10000, bins=100):
        """
        Estimate the PDF and CDF of the transition kernel P(X_{t+n} | X_t = x0)
        using histogram and empirical CDF.
        """
        self.reset_to_state(x0, num_samples)
        self.simulate(n_steps, num_chains=num_samples, reset=False)
        final_states = self.path[:, -1]

        hist_counts, bin_edges = np.histogram(final_states, bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        def pdf_func(x):
            return np.interp(x, bin_centers, hist_counts, left=0, right=0)

        sorted_samples = np.sort(final_states)
        def cdf_func(x):
            x = np.array([x])
            out = np.searchsorted(sorted_samples, x, side="right") / num_samples
            return out[0,:]

        return pdf_func, cdf_func, bin_centers

    def plot_transition_kernel(self, x0, n_steps=1, num_samples=10000, bins=100):
        pdf, cdf, bin_centers = self.estimate_transition_kernel(x0, n_steps, num_samples, bins)
        x_vals = np.linspace(bin_centers[0], bin_centers[-1], 300)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(x_vals, pdf(x_vals), color='blue')
        plt.title(f"PDF of P(X(t+{n_steps}) | X(t)={x0})")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(x_vals, cdf(x_vals), color='green')
        plt.title(f"CDF of P(X(t+{n_steps}) | X(t)={x0})")
        plt.xlabel("x")
        plt.ylabel("Probability")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


phi = 1


mc = AR1Model(phi=phi,initial_std=0.0)

path = mc.simulate(1000,1000)

mc.plot_path()
mc.plot_empirical_distribution()
mc.plot_average_distribution()




mc.plot_transition_kernel(0.0, num_samples=10**6)


from mpl_toolkits.mplot3d import Axes3D
