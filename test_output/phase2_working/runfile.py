import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

data_files = os.listdir(working_dir)

npz_path = os.path.join(working_dir, "experiment_data.npz")
if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    exp_data = data["experiment_data"].item()

    if "scenarios" in exp_data:
        scenarios = exp_data["scenarios"]
        is_single_condition = len(scenarios) == 1 and "default" in scenarios

        if is_single_condition:
            scenario_data = scenarios["default"]
            metrics = scenario_data.get("metrics", {})
            events = scenario_data.get("events", [])

            try:
                plt.figure()
                plt.bar(metrics.keys(), metrics.values())
                plt.title("Metrics Overview")
                plt.savefig(os.path.join(working_dir, "metrics_overview.png"))
                plt.close()
            except Exception as e:
                print(f"Error creating plot: {e}")
                plt.close()

        else:
            for scenario_name, scenario_data in scenarios.items():
                metrics = scenario_data.get("metrics", {})
                try:
                    plt.figure()
                    plt.bar(metrics.keys(), metrics.values())
                    plt.title(f"{scenario_name} - Metrics Overview")
                    plt.savefig(
                        os.path.join(
                            working_dir, f"{scenario_name}_metrics_overview.png"
                        )
                    )
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot for {scenario_name}: {e}")
                    plt.close()

                if "messages" in scenario_data:
                    # Assuming messages contain time-series data
                    message_counts = [
                        len(message) for message in scenario_data["messages"]
                    ]
                    try:
                        plt.figure()
                        plt.plot(message_counts)
                        plt.title(f"{scenario_name} - Message Counts Over Rounds")
                        plt.xlabel("Round")
                        plt.ylabel("Message Count")
                        plt.savefig(
                            os.path.join(
                                working_dir, f"{scenario_name}_message_counts.png"
                            )
                        )
                        plt.close()
                    except Exception as e:
                        print(
                            f"Error creating message count plot for {scenario_name}: {e}"
                        )
                        plt.close()
