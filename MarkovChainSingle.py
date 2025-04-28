import numpy as np
import matplotlib.pyplot as plt


class MarkovChainSingle:
    def __init__(self, initial_distribution, transition_kernel):
        """
        initial_distribution: 1D numpy array of shape (n,)
        transition_kernel: 2D numpy array of shape (n, n)
        """
        self.initial_distribution = np.array(initial_distribution)
        self.transition_kernel = np.array(transition_kernel)
        self.num_states = self.initial_distribution.shape[0]
        self.current_state = None
        self.path = None

    def reset(self):
        """
        Reset the chain to start a new simulation.
        """
        self.current_state = np.random.choice(self.num_states, p=self.initial_distribution)
        self.path = [self.current_state]

    def step(self):
        """
        Perform one step of the Markov chain.
        """
        probs = self.transition_kernel[self.current_state]
        next_state = np.random.choice(self.num_states, p=probs)
        self.current_state = next_state
        self.path.append(self.current_state)

    def simulate(self, steps, reset=True):
        """
        Simulate the Markov chain for a given number of steps.

        Args:
            steps: Number of steps to simulate.

        Returns:
            Numpy array of visited states.
        """
        if reset:
            self.reset()
        else:
            self.path = list(self.path)
        for _ in range(steps):
            self.step()
        self.path = np.array(self.path)
        return self.path

    def plot_path(self):
        """
        Plot the sequence of visited states over time.
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        plt.figure(figsize=(10, 4))
        plt.plot(range(len(self.path)), self.path, marker='o', linestyle='-', color='blue')
        plt.yticks(range(self.num_states))
        plt.xlabel('Time step')
        plt.ylabel('State')
        plt.title('Markov Chain Path')
        plt.grid(True)
        plt.show()

    def plot_empirical_distribution(self):
        """
        Plot the empirical distribution of visited states.
        """
        if self.path is None:
            raise ValueError("No path found. Run simulate() first.")

        counts = np.bincount(self.path, minlength=self.num_states)
        probs = counts / len(self.path)

        plt.figure(figsize=(6, 4))
        plt.bar(range(self.num_states), probs, color='skyblue')
        plt.xlabel('State')
        plt.ylabel('Empirical Probability')
        plt.title('Empirical Distribution')
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

    mc = MarkovChainSingle(initial_distribution, transition_kernel)

    # Simulate the chain
    path = mc.simulate(steps=50)
    print("Simulated path:", path)

    # Plot results
    mc.plot_path()
    mc.plot_empirical_distribution()
