import random

def markov_chain_monte_carlo(weather_states, transition_matrix, initial_state, steps):
    """
    Simulate weather probabilities using Markov Chain Monte Carlo.

    :param weather_states: List of possible weather states (e.g., ["Sunny", "Rainy", "Cloudy"])
    :param transition_matrix: Matrix of transition probabilities between states
    :param initial_state: Initial weather state
    :param steps: Number of steps to simulate
    :return: Dictionary with probabilities of each weather state
    """
    current_state = initial_state
    state_counts = {state: 0 for state in weather_states}

    for _ in range(steps):
        state_counts[current_state] += 1
        next_state = random.choices(
            weather_states, weights=transition_matrix[weather_states.index(current_state)]
        )[0]
        current_state = next_state

    total_steps = sum(state_counts.values())
    probabilities = {state: count / total_steps for state, count in state_counts.items()}
    return probabilities


if __name__ == "__main__":
    # Define weather states and transition probabilities
    weather_states = ["Sunny", "Rainy", "Cloudy"]
    transition_matrix = [
        [0.7, 0.2, 0.1],  # Probabilities from "Sunny"
        [0.3, 0.4, 0.3],  # Probabilities from "Rainy"
        [0.2, 0.3, 0.5],  # Probabilities from "Cloudy"
    ]

    # Initial state and number of steps
    initial_state = "Sunny"
    steps = 10000

    # Run the simulation
    probabilities = markov_chain_monte_carlo(weather_states, transition_matrix, initial_state, steps)

    # Print the results
    print("Weather probabilities after simulation:")
    for state, prob in probabilities.items():
        print(f"{state}: {prob:.4f}")