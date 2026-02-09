import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    runs = [d for d in os.listdir("outputs") if os.path.isdir(f"outputs/{d}")]
    print("\n".join([f"{i}: {r}" for i, r in enumerate(runs)]))
    choice = int(input("Select Run Index: "))
    run_path = f"outputs/{runs[choice]}"
    
    df = pd.read_csv(f"{run_path}/logs/telemetry_log.csv")
    os.makedirs(f"{run_path}/plots", exist_ok=True)

    # 1. Position Heatmap
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x="pos_x", y="pos_z", fill=True, cmap="viridis")
    plt.title("Track Position Density")
    plt.savefig(f"{run_path}/plots/heatmap.png")

    # 2. Action Frequency
    plt.figure()
    df['action'].value_counts().sort_index().plot(kind='bar')
    plt.title("Action Distribution")
    plt.savefig(f"{run_path}/plots/actions.png")

    # 3. Terminal Reasons
    plt.figure()
    df[df['reason'].notna()]['reason'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Episode Termination Reasons")
    plt.savefig(f"{run_path}/plots/reasons.png")
    
    print(f"Plots saved to {run_path}/plots/")

if __name__ == "__main__":
    generate_plots()