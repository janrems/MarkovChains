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

    def reset(self, num_chains=1):
        """
        Reset the Markov chain(s) with new initial states.

        Args:
            num_chains: number of independent chains to initialize
        """
        self.current_state = np.random.choice(self.num_states, size=num_chains, p=self.initial_distribution)
        self.path = [self.current_state.copy()]  # Start new path

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
    mc.simulate(steps=50, num_chains=2)
    print("Single chain path shape:", mc.path.shape)
    mc.plot_path()
    mc.plot_empirical_distribution()

    # Simulate multiple chains
    mc.simulate(steps=100, num_chains=500)
    print("Multiple chains path shape:", mc.path.shape)
    mc.plot_average_distribution()


