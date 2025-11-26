import os
import numpy as np
from autogen import Agent, MultiAgent, scenario

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)


class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.tokens = 10
        self.past_contributions = []
        self.score = 0

    def decide_contribution(self, rule=None):
        if rule:
            return rule[self.player_id]
        return np.random.randint(0, self.tokens + 1)


def run_simulation(scenario_name, rounds, rule=None, seeds=1):
    results = []
    for seed in range(seeds):
        np.random.seed(seed)
        players = [Player(i) for i in range(4)]
        round_logs = []

        for round_number in range(rounds):
            contributions = [player.decide_contribution(rule) for player in players]
            total_contribution = sum(contributions)
            threshold = rule[-1] if rule else 22

            for player, contribution in zip(players, contributions):
                if total_contribution >= threshold:
                    player.score = (
                        10
                        - contribution
                        + (1 if total_contribution >= threshold else 0)
                    )
                else:
                    player.score = 10 - contribution
                player.past_contributions.append(contribution)

            round_logs.append(
                {
                    "round_number": round_number + 1,
                    "contributions": contributions,
                    "total": total_contribution,
                    "threshold_met": total_contribution >= threshold,
                    "scores": [player.score for player in players],
                }
            )

        results.append(round_logs)

    return results


# Simulation scenarios
scenarios = {
    "FAIRSUFF": (20, [5, 5, 5, 5]),
    "UNFAIRSUFF": (22, [5, 5, 6, 6]),
    "CONTROL": (22, None),
}

experiment_data = {}

for scenario_name, (threshold, rule) in scenarios.items():
    runs = run_simulation(scenario_name, 20, rule, seeds=11)
    experiment_data[scenario_name] = {"runs": runs, "aggregated_metrics": {}}

# Save experiment data
np.savez_compressed(os.path.join(working_dir, "experiment_data.npz"), **experiment_data)
