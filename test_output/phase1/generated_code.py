import os
import random
import numpy as np

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


class Agent:
    def __init__(self, id):
        self.id = id
        self.last_contribution = 0
        self.memory = []

    def decide_contribution(self, total_tokens, target, round_number):
        # Make a simple decision based on target, previous contributions, etc.
        if round_number <= 10:  # No rules in the first half
            return random.randint(0, total_tokens)
        else:  # Apply rules based on the scenario for the second half
            return max(
                0, target[self.id] - self.last_contribution
            )  # Try to match the rule


def run_simulation(scenario):
    agents = [Agent(i) for i in range(4)]
    total_rounds = 20
    metrics = {
        "rounds": [],
        "contributions": [],
        "total_contributed": [],
        "threshold_met": [],
        "individual_scores": [],
    }

    for round_number in range(total_rounds):
        contributions = [
            agent.decide_contribution(10, scenario["rules"], round_number)
            for agent in agents
        ]
        total_contribution = sum(contributions)
        threshold_met = total_contribution >= scenario["threshold"]

        # Update scores
        for i, agent in enumerate(agents):
            if threshold_met:
                agent_score = 10 - contributions[i] + scenario["reward"]
            else:
                agent_score = 10 - contributions[i]
            agent.last_contribution = contributions[i]
            metrics["individual_scores"].append(agent_score)

        # Collect data for analysis
        metrics["rounds"].append(round_number + 1)
        metrics["contributions"].append(contributions)
        metrics["total_contributed"].append(total_contribution)
        metrics["threshold_met"].append(threshold_met)

    return metrics


def main():
    scenarios = [
        {"name": "FAIRSUFF", "threshold": 20, "rules": [5, 5, 5, 5], "reward": 10},
        {"name": "UNFAIRINF", "threshold": 22, "rules": [6, 6, 6, 6], "reward": 10},
    ]

    experiment_data = {}

    for scenario in scenarios:
        logs = {"runs": []}
        for seed in range(3):  # Running multiple trials with different seeds
            random.seed(seed)
            metrics = run_simulation(scenario)
            logs["runs"].append({"seed": seed, "metrics": metrics})

        experiment_data[scenario["name"]] = logs

    # Save experiment data for later analysis
    np.savez_compressed(
        os.path.join(working_dir, "experiment_data.npz"), experiment_data
    )


for run_metrics in experiment_data.values():
    for run in run_metrics["runs"]:
        print(f'Run with seed={run["seed"]}: metrics = {run["metrics"]}')
