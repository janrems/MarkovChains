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




mc.plot_transition_kernel(3.0, num_samples=10**6)


##########

from mpl_toolkits.mplot3d import Axes3D

class AR1Multivariate:
    def __init__(self, phi_matrix, noise_cov=None, initial_mean=None, initial_cov=None):
        self.phi = np.array(phi_matrix)
        self.d = self.phi.shape[0]

        self.noise_cov = np.eye(self.d) if noise_cov is None else np.array(noise_cov)
        self.initial_mean = np.zeros(self.d) if initial_mean is None else np.array(initial_mean)
        self.initial_cov = np.eye(self.d) if initial_cov is None else np.array(initial_cov)

        self.current_state = None
        self.path = None

    def reset(self, num_chains=1):
        self.current_state = np.random.multivariate_normal(self.initial_mean, self.initial_cov, size=num_chains)
        self.path = [self.current_state.copy()]

    def reset_to_state(self, state, num_chains=1):
        self.current_state = np.tile(np.array(state), (num_chains, 1))
        self.path = [self.current_state.copy()]

    def step(self):
        noise = np.random.multivariate_normal(np.zeros(self.d), self.noise_cov, size=self.current_state.shape[0])
        self.current_state = self.current_state @ self.phi.T + noise
        self.path.append(self.current_state.copy())

    def simulate(self, steps, num_chains=1, reset=True):
        if reset:
            self.reset(num_chains)
        else:
            self.path = list(self.path)

        for _ in range(steps):
            self.step()

        self.path = np.stack(self.path, axis=1)
        return self.path

    def plot_path(self, chain_index=0):
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        path = self.path[chain_index]  # shape (steps+1, d)

        if self.d == 1:
            plt.plot(path[:, 0], marker='o', linestyle='-')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('1D AR(1) Path')
            plt.grid(True)
            plt.show()

        elif self.d == 2:
            plt.plot(path[:, 0], path[:, 1], marker='o', linestyle='-')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('2D AR(1) Path')
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        elif self.d == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('X3')
            ax.set_title('3D AR(1) Path')
            plt.show()

        else:
            raise NotImplementedError("Path plotting is only supported for dimensions <= 3")

    def plot_empirical_distribution(self, chain_index=0, bins=30):
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        path = self.path[chain_index]  # (steps+1, d)

        if self.d == 1:
            plt.hist(path[:, 0], bins=bins, density=True, color='skyblue')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Empirical Distribution (Chain {chain_index})')
            plt.grid(True)
            plt.show()

        elif self.d == 2:
            plt.hist2d(path[:, 0], path[:, 1], bins=bins, density=True, cmap='Blues')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title(f'2D Empirical Distribution (Chain {chain_index})')
            plt.colorbar(label='Density')
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        else:
            raise NotImplementedError("Empirical distribution plotting supported only for d <= 2")

    def plot_average_distribution(self, bins=30):
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        values = self.path.reshape(-1, self.d)  # (num_chains * (steps+1), d)

        if self.d == 1:
            plt.hist(values[:, 0], bins=bins, density=True, color='lightgreen')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title('Average Empirical Distribution')
            plt.grid(True)
            plt.show()

        elif self.d == 2:
            plt.hist2d(values[:, 0], values[:, 1], bins=bins, density=True, cmap='Greens')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('2D Average Empirical Distribution')
            plt.colorbar(label='Density')
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        else:
            raise NotImplementedError("Average distribution plotting supported only for d <= 2")

    def plot_transition_kernel(self, x0, n_steps=1, num_samples=10000, bins=100):
        x0 = np.array(x0)
        if x0.ndim == 0:
            x0 = x0[None]

        self.reset_to_state(x0, num_samples)
        self.simulate(n_steps, num_chains=num_samples, reset=False)
        final_states = self.path[:, -1]  # shape (num_samples, d)

        if self.d == 1:
            plt.figure(figsize=(6, 4))
            plt.hist(final_states[:, 0], bins=bins, density=True, color='orange')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f'Transition Kernel PDF after {n_steps} steps from x0={x0[0]}')
            plt.grid(True)
            plt.show()

        elif self.d == 2:
            plt.figure(figsize=(6, 6))
            plt.hist2d(final_states[:, 0], final_states[:, 1], bins=bins, density=True, cmap='Oranges')
            plt.xlabel('Dim 1')
            plt.ylabel('Dim 2')
            plt.title(f'Transition Kernel PDF after {n_steps} steps from x0={x0}')
            plt.colorbar(label='Density')
            plt.grid(True)
            plt.show()

        else:
            raise ValueError("Transition kernel plotting supported only for 1D or 2D processes.")


phi = np.array([[0.7, 0.1],
                [0.0, 0.9]])



mc2 = AR1Multivariate(phi)
path = mc2.simulate(steps=20, num_chains=1000, reset=True)
mc2.plot_path()

# Plot PDF of 2D transition kernel
mc2.plot_transition_kernel(x0=[1.0, 0.0], n_steps=5)
#mc2.plot_empirical_distribution()
mc2.plot_average_distribution()

mc2.plot_transition_kernel(x0=[1.0, 0.0])






















