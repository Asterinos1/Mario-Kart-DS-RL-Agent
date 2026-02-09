import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tf_logs(run_path, run_name):
    """Extracts scalar data from a specific run directory."""
    data = {}
    target_tags = ['rollout/ep_rew_mean', 'train/loss', 'rollout/ep_len_mean', 'train/learning_rate']
    
    # Search for tfevents inside the specific run folder
    for root, _, files in os.walk(run_path):
        for file in files:
            if "tfevents" in file:
                ea = EventAccumulator(os.path.join(root, file))
                ea.Reload()
                available_tags = ea.Tags()['scalars']
                
                for tag in target_tags:
                    if tag in available_tags:
                        events = ea.Scalars(tag)
                        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
                        df['run'] = run_name
                        if tag not in data: data[tag] = []
                        data[tag].append(df)
    return data

def save_plots(all_data, save_base_dir, is_comparison=False):
    """Renders prettier plots with grids and specific color palettes."""
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(save_base_dir, exist_ok=True)

    for tag, df_list in all_data.items():
        if not df_list: continue
        
        plt.figure(figsize=(10, 6))
        combined_df = pd.concat(df_list)
        clean_name = tag.replace("/", "_")
        
        # Plotting with smoothed lines if multiple runs exist
        sns.lineplot(data=combined_df, x="step", y="value", hue="run", linewidth=2, alpha=0.8)
        
        plt.title(f"Metric: {tag.split('/')[-1].replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        
        if is_comparison:
            plt.legend(title="Runs", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = f"tf_{clean_name}.png"
        plt.savefig(os.path.join(save_base_dir, filename), dpi=300)
        plt.close()
    print(f"Done. Plots saved to: {save_base_dir}")

def run_menu():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    log_dir = ROOT_DIR / "logs"
    output_dir = ROOT_DIR / "outputs"

    if not log_dir.exists():
        print(f"Error: Directory '{log_dir}' not found. No runs to examine.")
        return

    runs = [d for d in os.listdir(log_dir) if os.path.isdir(log_dir / d)]
    if not runs:
        print("No runs found in logs directory.")
        return

    print("\n--- TensorBoard Log Analysis ---")
    print("0: [PLOT ALL RUNS TOGETHER]")
    for i, run in enumerate(runs, 1):
        print(f"{i}: {run}")
    
    try:
        choice = int(input("\nSelect Run Index: "))
        
        if choice == 0:
            # Multi-run comparison
            all_data = {}
            for r in runs:
                run_data = extract_tf_logs(str(log_dir / r), r)
                for tag, dfs in run_data.items():
                    if tag not in all_data: all_data[tag] = []
                    all_data[tag].extend(dfs)
            
            # Save to analysis/plots/comparison
            save_path = ROOT_DIR / "analysis" / "plots" / "comparison"
            save_plots(all_data, str(save_path), is_comparison=True)
            
        elif 1 <= choice <= len(runs):
            selected_run = runs[choice - 1]
            # Strip trailing _0 often added by SB3 to find matching output folder
            base_run_name = selected_run.rsplit('_', 1)[0] if selected_run.endswith('_0') else selected_run
            
            run_data = extract_tf_logs(str(log_dir / selected_run), selected_run)
            
            # Target the specific output/plots folder
            save_path = output_dir / base_run_name / "plots"
            save_plots(run_data, str(save_path), is_comparison=False)
        else:
            print("Invalid selection.")
    except (ValueError, IndexError):
        print("Invalid input.")

if __name__ == "__main__":
    run_menu()