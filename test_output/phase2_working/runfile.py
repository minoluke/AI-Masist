import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Discover and load data files
for filename in os.listdir(working_dir):
    if filename.endswith(".npz"):
        try:
            data = np.load(os.path.join(working_dir, filename), allow_pickle=True)
            total_messages = data["total_messages"].item()
            scenario_name = filename[:-4]  # Remove .npz extension

            # Time-series plot of total contributions over rounds
            rounds = [result[0] for result in total_messages]
            contributions = [result[2] for result in total_messages]
            plt.figure()
            plt.plot(rounds, contributions, marker="o")
            plt.title(f"Total Contributions Over Rounds for {scenario_name}")
            plt.xlabel("Round")
            plt.ylabel("Total Contributions")
            plt.savefig(
                os.path.join(working_dir, f"{scenario_name}_total_contributions.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    elif filename.endswith(".npy"):
        try:
            data = np.load(os.path.join(working_dir, filename), allow_pickle=True)
            scenario_name = filename[:-4]  # Remove .npy extension

            # Process and create a simple distribution plot
            contributions = data  # Assuming this holds the contributions data
            plt.figure()
            plt.hist(contributions, bins=10, alpha=0.7)
            plt.title(f"Contributions Distribution for {scenario_name}")
            plt.xlabel("Contribution Amount")
            plt.ylabel("Frequency")
            plt.savefig(
                os.path.join(
                    working_dir, f"{scenario_name}_contribution_distribution.png"
                )
            )
            plt.close()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
