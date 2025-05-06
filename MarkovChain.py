import numpy as np
import matplotlib.pyplot as plt


class MarkovChain:
    def __init__(self, initial_distribution, transition_kernel):
        """
        initial_distribution: 1D numpy array of shape (n,)
        transition_kernel: 2D numpy array of shape (n, n)
        """
        self.initial_distribution = np.array(initial_distribution)
        self.transition_kernel = np.array(transition_kernel)
        self.num_states = self.initial_distribution.shape[0]
        self.path = None  # Will store path here later
        self.current_state = None  # Current state
        self.test_probs()

    def test_probs(self):
        if np.sum(self.initial_distribution) != 1:
            raise ValueError("Initial distribution doesn't sum up to 1")

        row_sums = np.sum(self.transition_kernel, axis=1)
        if not np.prod(row_sums==1):
            raise ValueError("Transition Kernel is not a stochastic matrix")

    def reset(self, num_chains=1):
        """
        Reset the Markov chain(s) with new initial states, from initial distribution

        Args:
            num_chains: number of independent chains to initialize
        """
        self.current_state = np.random.choice(self.num_states, size=num_chains, p=self.initial_distribution)
        self.path = [self.current_state.copy()]  # Start new path

    def reset_to_state(self, state, num_chains = 1):
        """
        Reset the Markov chain(s) with new initial states, from initial distribution

        Args:
            state: initial state
            num_chains: number of independent chains to initialize
        """
        self.current_state = np.ones(num_chains, dtype=np.int64)*state
        self.path = [self.current_state.copy()]

    def step_for(self):
        """
        Perform one step of the Markov chain(s).
        """
        probs = self.transition_kernel[self.current_state]
        if probs.ndim == 1:
            # Single chain
            next_state = np.random.choice(self.num_states, p=probs)
        else:
            # Multiple chains
            next_state = np.array([
                np.random.choice(self.num_states, p=probs[i]) for i in range(probs.shape[0])
            ])
        self.current_state = next_state
        self.path.append(self.current_state.copy())

    def step(self):
        """
        Perform one step of the Markov chain(s) in a fully vectorized way.
        """
        probs = self.transition_kernel[self.current_state]
        if probs.ndim == 1:
            next_state = np.random.choice(self.num_states, p=probs)
        else:
            rand_vals = np.random.rand(probs.shape[0])
            cumulative_probs = np.cumsum(probs, axis=1)
            next_state = (rand_vals[:, None] < cumulative_probs).argmax(axis=1)

        self.current_state = next_state
        self.path.append(self.current_state.copy())

    def simulate(self, steps, num_chains=1, reset=True):
        """
        Simulate the Markov chain for a given number of steps.

        Args:
            steps: Number of steps to simulate
            num_chains: Number of chains to simulate

        Returns:
            numpy array of shape (num_chains, steps+1)
        """
        if reset:
            self.reset(num_chains=num_chains)
        else:
            self.path = list(self.path)
        for _ in range(steps):
            self.step()
        # Convert path to numpy array for easier processing
        self.path = np.stack(self.path, axis=1)  # shape (num_chains, steps+1)


    def plot_path(self, chain_index=0):
        """
        Plot the path of a single chain.

        Args:
            chain_index: which chain to plot (default first one)
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        path = self.path[chain_index]
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(path)), path, marker='o', linestyle='-', color='blue')
        plt.yticks(range(self.num_states))
        plt.xlabel('Time step')
        plt.ylabel('State')
        plt.title(f'Markov Chain Path (Chain {chain_index})')
        plt.grid(True)
        plt.show()

    def plot_empirical_distribution(self, chain_index=0):
        """
        Plot the empirical distribution of states for a single chain.

        Args:
            chain_index: which chain to analyze (default first one)
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        path = self.path[chain_index]
        counts = np.bincount(path, minlength=self.num_states)
        probs = counts / len(path)

        plt.figure(figsize=(6, 4))
        plt.bar(range(self.num_states), probs, color='skyblue')
        plt.xlabel('State')
        plt.ylabel('Empirical Probability')
        plt.title(f'Empirical Distribution (Chain {chain_index})')
        plt.xticks(range(self.num_states))
        plt.grid(axis='y')
        plt.show()

    def plot_average_distribution(self):
        """
        Plot the average empirical distribution over all chains.
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        all_counts = np.bincount(self.path.flatten(), minlength=self.num_states)
        avg_probs = all_counts / self.path.size

        plt.figure(figsize=(6, 4))
        plt.bar(range(self.num_states), avg_probs, color='lightgreen')
        plt.xlabel('State')
        plt.ylabel('Average Empirical Probability')
        plt.title('Average Distribution Over All Chains')
        plt.xticks(range(self.num_states))
        plt.grid(axis='y')
        plt.show()



    def n_step_kernel(self, n):
        """
        Compute the n-step transition kernel.

        Args:
            n: Number of steps

        Returns:
            2D numpy array of shape (num_states, num_states), the n-step transition matrix.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        return np.linalg.matrix_power(self.transition_kernel, n)


    def estimate_transition_kernel(self, n_steps, num_smaples=1000):
        """
        Empirically estimate the n-step transition kernel by simulating transitions.

        Args:
            n_steps: Number of steps for each simulation.
            num_smaples: Number of simulations per starting state.

        Returns:
            2D numpy array of shape (num_states, num_states) estimating the n-step transition kernel.
        """
        counts = np.zeros((self.num_states, self.num_states), dtype=np.int64)

        for start_state in range(self.num_states):
            # Manually set the current_state to all be `start_state`
            self.reset_to_state(start_state,num_smaples)

            # Now use `simulate` to run n_steps (without resetting!)
            self.simulate(n_steps, num_chains=num_smaples, reset=False)

            # Get the final states after n_steps
            final_states = self.path[:, -1]

            # Count transitions to each end_state
            for end_state in range(self.num_states):
                counts[start_state, end_state] = np.sum(final_states == end_state)

        # Normalize counts to get probabilities
        empirical_kernel = counts / num_smaples
        return empirical_kernel

    def compare_kernels(self, n_steps, num_samples_per_state=1000):
        """
        Compare the empirical and theoretical n-step transition kernels.

        Args:
            n_steps: Number of steps.
            num_samples_per_state: Simulations per state for empirical estimate.
            verbose: Whether to print matrices and differences.

        Returns:
            A dictionary with keys: 'empirical', 'theoretical', 'difference'
        """
        empirical = self.estimate_transition_kernel(n_steps, num_samples_per_state)
        theoretical = self.n_step_kernel(n_steps)
        difference = np.abs(empirical - theoretical)


        plt.figure(figsize=(6, 5))
        plt.imshow(difference, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Absolute Difference')
        plt.xlabel('To state')
        plt.ylabel('From state')
        plt.title('Empirical vs Theoretical Difference')
        plt.show()

    def stationary_distribution(self):
        """
        Compute the stationary distribution of the Markov chain.

        Returns:
            1D numpy array of shape (num_states,) representing the stationary distribution.
        """
        eigvals, eigvecs = np.linalg.eig(self.transition_kernel.T)
        # Find the eigenvector corresponding to eigenvalue 1
        index = np.argmin(np.abs(eigvals - 1))
        stationary = np.real(eigvecs[:, index])
        # Normalize to sum to 1
        stationary_distribution = stationary / np.sum(stationary)
        return stationary_distribution

    def compute_mixing_time(self, tolerance=1e-2, max_steps=1000, num_chains=1000, to_print=True):
        """
        Estimate how many steps are needed for the empirical distribution to approach the stationary distribution.

        Args:
            tolerance: Threshold for convergence (e.g., total variation distance).
            max_steps: Maximum number of steps to simulate.
            num_chains: Number of chains to run in parallel.

        Returns:
            steps_to_convergence: Number of steps where empirical distribution is within tolerance of stationary.
        """
        stationary = self.stationary_distribution()
        self.reset(num_chains)  # Initialize chains

        for step in range(1, max_steps + 1):
            self.step()

            # Get empirical distribution across all chains at this step
            counts = np.bincount(self.current_state, minlength=self.num_states)
            empirical = counts / num_chains

            # Compute total variation distance (L1 norm / 2)
            tv_distance = np.sum(np.abs(empirical - stationary))


            if tv_distance < tolerance:
                return step

        if to_print:
            print(f"Did not converge within {max_steps} steps (final TV distance: {tv_distance:.5f})")
        return None  # Or max_steps if you want to return the upper limit

    def expected_mixing_time(self, sample_size=100, tolerance=1e-2, max_steps=1000, num_chains=1000):
        sum = 0
        nones = 0
        for i in range(sample_size):
            time = self.compute_mixing_time(tolerance,max_steps,num_chains, to_print=False)
            if time != None:
                sum += time
            else:
                nones += 1

        if nones==0:
            print("Mixing time always reached")
            return sum/sample_size
        else:
            sum += nones*max_steps
            print(f"Mixing time at least {sum/sample_size}")
            print(f"\nMixing time not reached {nones} times")
            return sum/sample_size


# Example usage
if __name__ == "__main__":
    initial_distribution = [0.2, 0.5, 0.3]
    transition_kernel = [
        [0.5, 0.3, 0.2],
        [0.1, 0.6, 0.3],
        [0.2, 0.4, 0.4]
    ]

    mc = MarkovChain(initial_distribution, transition_kernel)

    # Simulate a single chain
    mc.simulate(steps=50, num_chains=1)
    print("Single chain path shape:", mc.path.shape)
    mc.plot_path()
    mc.plot_empirical_distribution()

    # Simulate multiple chains
    mc.simulate(steps=100, num_chains=2)
    print("Multiple chains path shape:", mc.path.shape)
    mc.plot_average_distribution()

    mc.n_step_kernel(10)

    mc.estimate_transition_kernel(10)


    mc.compare_kernels(10,100000)

    mc.stationary_distribution()

    mc.compute_mixing_time()

    mc.expected_mixing_time(tolerance=0.001)







