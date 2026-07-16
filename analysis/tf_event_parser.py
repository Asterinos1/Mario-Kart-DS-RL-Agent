"""TensorBoard log parser and visualizer for Mario Kart DS RL training runs.

Extracts scalar metrics from TensorFlow event files (``tfevents.*``) produced
by Stable-Baselines3 and generates high-fidelity training plots for either a
single selected run or a multi-run overlay comparison.

Typical workflow::

    python analysis/tf_event_parser.py

    --- TensorBoard Log Analysis ---
    0: [PLOT ALL RUNS TOGETHER]
    1: run_20240101_120000
    2: run_20240102_090000
    Select Run Index: 0   # overlays all runs on each metric plot
    Select Run Index: 1   # plots only run_20240101_120000

Metrics parsed (when present in event files):
    - rollout/ep_rew_mean      – mean episode reward
    - rollout/ep_len_mean      – mean episode length
    - rollout/exploration_rate – epsilon value during exploration decay
    - rollout/fps              – environment frames per second (rollout phase)
    - train/learning_rate      – current learning rate
    - train/loss               – TD / policy loss
    - train/n_updates          – cumulative gradient update count
    - time/fps                 – overall training throughput in fps
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_tf_logs(run_path, run_name):
    """Extract scalar metric data from TensorFlow event files in a run directory.

    Walks the entire ``run_path`` subtree looking for files whose name contains
    ``"tfevents"``.  For each event file found, loads it via
    :class:`~tensorboard.backend.event_processing.event_accumulator.EventAccumulator`
    and extracts only the scalar tags listed in ``target_tags``.

    Args:
        run_path (str): Absolute path to the run directory to search.  The
            function walks recursively, so nested subdirectories are included.
        run_name (str): Human-readable label assigned to every DataFrame row
            via the ``run`` column.  Used as the ``hue`` identifier when
            overlaying multiple runs in :func:`save_plots`.

    Returns:
        dict[str, list[pd.DataFrame]]: A mapping from metric tag name to a
        list of DataFrames.  Each DataFrame has three columns:

        - ``step``  (int)   – global training step at which the scalar was
          recorded.
        - ``value`` (float) – scalar value at that step.
        - ``run``   (str)   – copy of ``run_name`` on every row, enabling
          ``hue``-based grouping in seaborn.

        If a tag was not found in any event file the key is absent from the
        returned dict.  If the run directory contains no event files an empty
        dict is returned.

    Example::

        data = extract_tf_logs("/outputs/run_A", "run_A")
        # data["rollout/ep_rew_mean"] -> [DataFrame with step/value/run cols]
    """
    data = {}
    #target_tags = ['rollout/ep_rew_mean', 'train/loss', 'rollout/ep_len_mean', 'train/learning_rate']
    target_tags = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
        'rollout/exploration_rate',  # Added to track epsilon decay
        'rollout/fps',
        'train/learning_rate',
        'train/loss',
        'train/n_updates',
        'time/fps',
    ]

    # Search for tfevents inside the specific run folder
    for root, _, files in os.walk(run_path):
        for file in files:
            if "tfevents" in file:
                # Load and parse the binary event file.
                ea = EventAccumulator(os.path.join(root, file))
                ea.Reload()
                # Tags() returns a dict keyed by data type; 'scalars' is a list
                # of tag names for which scalar events exist in this file.
                available_tags = ea.Tags()['scalars']

                for tag in target_tags:
                    if tag in available_tags:
                        events = ea.Scalars(tag)
                        # Each ScalarEvent namedtuple has .step, .value, .wall_time;
                        # we only need step and value for plotting.
                        df = pd.DataFrame([(e.step, e.value) for e in events], columns=['step', 'value'])
                        df['run'] = run_name  # Tag rows for hue-based multi-run coloring.
                        if tag not in data: data[tag] = []
                        data[tag].append(df)
    return data


def save_plots(all_data, save_base_dir, is_comparison=False):
    """Render and save one PNG per metric tag to ``save_base_dir``.

    For each tag in ``all_data`` the DataFrames in its list are concatenated
    and passed to :func:`seaborn.lineplot`.  When ``is_comparison`` is
    ``True`` the ``run`` column drives the ``hue`` parameter, drawing a
    separate coloured line per run and placing a legend outside the axes.
    When ``False`` (single-run mode) the ``hue`` parameter still works but
    there is only one category, so the legend is suppressed.

    Args:
        all_data (dict[str, list[pd.DataFrame]]): Output from one or more
            :func:`extract_tf_logs` calls.  Keys are metric tag strings;
            values are lists of DataFrames each with ``step``, ``value``, and
            ``run`` columns.
        save_base_dir (str): Directory path where PNG files will be written.
            Created automatically if it does not already exist.
        is_comparison (bool): When ``True``, enables per-run ``hue`` colouring
            and places an anchored legend to the right of the axes so run
            names do not occlude the plot area.  Defaults to ``False``.

    Returns:
        None: Plots are written to disk; function prints the save path on
        completion.

    Note:
        File names follow the pattern ``tf_<tag_with_slashes_replaced>.png``
        (e.g. ``tf_rollout_ep_rew_mean.png``).

    Example::

        save_plots(all_data, "/outputs/run_A/plots", is_comparison=False)
        save_plots(all_data, "/analysis/plots/comparison", is_comparison=True)
    """
    # Use a consistent muted palette across all plots for visual coherence.
    sns.set_theme(style="whitegrid", palette="muted")
    os.makedirs(save_base_dir, exist_ok=True)

    for tag, df_list in all_data.items():
        if not df_list: continue

        plt.figure(figsize=(10, 6))
        # Merge all per-run DataFrames into one so seaborn can draw them
        # together and automatically assign hue colours per run label.
        combined_df = pd.concat(df_list)
        # Replace '/' with '_' to produce a safe filename component.
        clean_name = tag.replace("/", "_")

        # Plotting with smoothed lines if multiple runs exist.
        # hue="run" assigns a distinct colour to each unique run name;
        # seaborn also averages over duplicate steps within a run (CI band).
        sns.lineplot(data=combined_df, x="step", y="value", hue="run", linewidth=2, alpha=0.8)

        # Derive a readable title from the tag's leaf segment.
        plt.title(f"Metric: {tag.split('/')[-1].replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Value", fontsize=12)

        if is_comparison:
            # Anchor legend outside the plot area to avoid occluding data lines
            # when many runs are overlaid simultaneously.
            plt.legend(title="Runs", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        filename = f"tf_{clean_name}.png"
        plt.savefig(os.path.join(save_base_dir, filename), dpi=300)
        plt.close()
    print(f"Done. Plots saved to: {save_base_dir}")


def run_menu():
    """CLI entry point for interactive TensorBoard log analysis.

    Scans ``<project_root>/logs/`` for run subdirectories and presents a
    numbered menu.  Two operating modes are available based on the user's
    numeric choice:

    **Mode 0 – Compare all runs:**
        Iterates over every subdirectory in ``logs/``, calls
        :func:`extract_tf_logs` for each, merges all results into a single
        ``all_data`` dict, and calls :func:`save_plots` with
        ``is_comparison=True``.  Plots are saved to
        ``<project_root>/analysis/plots/comparison/``.

    **Mode N (1 … len(runs)) – Single run:**
        Processes only the run at index N.  If the directory name ends with
        ``_0`` (e.g. ``my_experiment_0``) that suffix is stripped to produce
        the ``base_run_name`` used as the output subdirectory, keeping the
        folder structure consistent with non-indexed runs.  Plots are saved
        to ``<project_root>/outputs/<base_run_name>/plots/``.

    Returns:
        None: Output is written to disk; the function prints confirmation
        messages and returns without a value.

    Raises:
        SystemExit: Not raised explicitly; invalid input prints a message and
            the function returns early instead.

    Note:
        The ``logs/`` directory is expected to contain SB3-style run folders
        that each hold at least one ``tfevents.*`` file (possibly nested inside
        a subdirectory named after the algorithm, e.g. ``DQN_1/``).

    Example::

        $ python analysis/tf_event_parser.py

        --- TensorBoard Log Analysis ---
        0: [PLOT ALL RUNS TOGETHER]
        1: dqn_run_20240101
        Select Run Index: 1
        Done. Plots saved to: .../outputs/dqn_run_20240101/plots
    """
    # Resolve paths relative to the repo root, not the cwd, for portability.
    ROOT_DIR = Path(__file__).resolve().parent.parent
    log_dir = ROOT_DIR / "logs"
    output_dir = ROOT_DIR / "outputs"

    if not log_dir.exists():
        print(f"Error: Directory '{log_dir}' not found. No runs to examine.")
        return

    # Collect only immediate child directories; each represents one training run.
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
            # -------------------------------------------------------------- #
            # Multi-run comparison: aggregate data from every run directory.  #
            # -------------------------------------------------------------- #
            all_data = {}
            for r in runs:
                run_data = extract_tf_logs(str(log_dir / r), r)
                # Merge this run's data into the shared dict by extending each
                # tag's DataFrame list (preserving the 'run' column per df).
                for tag, dfs in run_data.items():
                    if tag not in all_data: all_data[tag] = []
                    all_data[tag].extend(dfs)

            # Save to analysis/plots/comparison
            save_path = ROOT_DIR / "analysis" / "plots" / "comparison"
            save_plots(all_data, str(save_path), is_comparison=True)

        elif 1 <= choice <= len(runs):
            # -------------------------------------------------------------- #
            # Single-run mode: process only the selected run directory.       #
            # -------------------------------------------------------------- #
            selected_run = runs[choice - 1]
            # Strip the trailing '_0' index suffix (added by SB3 when a run
            # name is reused) so plots land in the canonical base run folder.
            base_run_name = selected_run.rsplit('_', 1)[0] if selected_run.endswith('_0') else selected_run

            run_data = extract_tf_logs(str(log_dir / selected_run), selected_run)

            save_path = output_dir / base_run_name / "plots"
            save_plots(run_data, str(save_path), is_comparison=False)
        else:
            print("Invalid selection.")
    except (ValueError, IndexError):
        print("Invalid input.")


if __name__ == "__main__":
    run_menu()