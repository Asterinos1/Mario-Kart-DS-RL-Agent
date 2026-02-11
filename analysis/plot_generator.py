import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots():
    """Generates performance plots from telemetry CSV logs."""
    sns.set_theme(style="whitegrid") 
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'figure.autolayout': True
    })

    base_dir = "../outputs"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' not found. Current path: {os.getcwd()}")
        return

    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not runs:
        print("No runs found in outputs directory.")
        return

    print("\n".join([f"{i}: {r}" for i, r in enumerate(runs)]))
    try:
        choice = int(input("Select Run Index: "))
        run_path = os.path.join(base_dir, runs[choice])
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    
    csv_path = os.path.join(run_path, "logs/telemetry_log.csv")
    df = pd.read_csv(csv_path)
    
    plot_dir = os.path.join(run_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Position Heatmap
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor('white')
    
    sns.kdeplot(
        data=df, x="pos_x", y="pos_z", 
        fill=True, 
        thresh=0.05, 
        levels=100, 
        cmap="mako",
        antialiased=True
    )
    
    plt.title("Track Position Density")
    plt.xlabel("Position X")
    plt.ylabel("Position Z")
    plt.grid(True) 
    plt.savefig(os.path.join(plot_dir, "heatmap.png"), dpi=300, facecolor='white')

    # 2. Action Frequency (With Descriptive Labels)
    plt.figure(figsize=(8, 6))
    action_map = {0: "Gas", 1: "Gas + Left", 2: "Gas + Right"}
    action_counts = df['action'].value_counts().sort_index()
    labels = [action_map.get(x, str(x)) for x in action_counts.index]
    sns.barplot(x=labels, y=action_counts.values, palette="viridis", hue=labels, legend=False)
    plt.title("Action Distribution")
    plt.xlabel("Action Type")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(plot_dir, "actions.png"), dpi=300)

    # 3. Terminal Reasons
    plt.figure(figsize=(8, 8))
    reason_counts = df[df['reason'].notna()]['reason'].value_counts()
    if not reason_counts.empty:
        plt.pie(
            reason_counts, 
            labels=reason_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("pastel"),
            wedgeprops={'edgecolor': 'white'}
        )
        plt.title("Episode Termination Reasons")
    plt.savefig(os.path.join(plot_dir, "reasons.png"), dpi=300)

    # 4. Speed vs Offroad Correlation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="offroad", y="speed", alpha=0.1, color='orange')
    plt.title("Speed vs. Offroad Performance")
    plt.xlabel("Offroad Modifier (Lower = More Grass)")
    plt.ylabel("Speed")
    plt.savefig(os.path.join(plot_dir, "speed_offroad.png"), dpi=300)
        
    print(f"Plots saved to {plot_dir}/")

if __name__ == "__main__":
    generate_plots()