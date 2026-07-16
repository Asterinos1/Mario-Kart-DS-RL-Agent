"""Stable-Baselines3 custom callbacks for Mario Kart DS RL training telemetry.

This module provides :class:`MKDSMetricsCallback`, a training callback that
captures per-step race telemetry (speed, off-road flag, position, chosen
action, reward, and terminal reason) from every parallel environment and
persists it to a CSV file.  Data are accumulated in an in-memory buffer and
flushed to disk in batches to avoid I/O overhead on every step.  The CSV is
opened in append mode so that interrupted training runs can be resumed without
losing previously collected data.
"""

import os
import csv
from stable_baselines3.common.callbacks import BaseCallback


class MKDSMetricsCallback(BaseCallback):
    """Custom callback for logging race telemetry to CSV during training.

    Hooks into the Stable-Baselines3 training loop to record per-step
    telemetry from every vectorised environment instance.  Rows are staged
    in :attr:`buffer` and written to disk in a single batch whenever the
    buffer reaches :attr:`flush_freq` entries, keeping per-step I/O cost
    negligible.  A final flush is performed when training ends so no data
    are silently discarded.

    Attributes:
        log_path (str): Absolute path to the output CSV file
            (``<log_dir>/telemetry_log.csv``).
        buffer (list[list]): In-memory staging area for telemetry rows
            awaiting the next batch write.
        flush_freq (int): Number of rows that must accumulate in
            :attr:`buffer` before an automatic flush to disk is triggered.
            Default is ``5000``.
    """

    def __init__(self, log_dir: str, verbose: int = 0) -> None:
        """Initialises the callback and resolves the output file path.

        Args:
            log_dir (str): Directory in which the telemetry CSV will be
                created.  The directory must already exist; the file itself
                is created (or appended to) on the first call to
                :meth:`_on_training_start`.
            verbose (int): Verbosity level passed to the parent
                :class:`~stable_baselines3.common.callbacks.BaseCallback`.
                ``0`` = silent, ``1`` = info, ``2`` = debug.  Defaults to
                ``0``.
        """
        super().__init__(verbose)

        # Build the full path once so every method can use self.log_path.
        self.log_path = os.path.join(log_dir, "telemetry_log.csv")

        # Rows accumulate here until flush_freq is reached.
        self.buffer = []

        # Flush to disk after this many buffered rows to balance RAM usage
        # against file-system write frequency.
        self.flush_freq = 5000

    def _on_training_start(self) -> None:
        """Initialises the CSV file header if the file does not yet exist.

        Opens the file in **append** mode (``'a'``) rather than write mode
        (``'w'``) so that a resumed training run continues adding rows to
        the same file instead of truncating data from previous sessions.
        The header row is written only when the file is new, preventing
        duplicate headers on resume.
        """
        # Check existence before opening so we can decide whether to write
        # the header without relying on the file position after open().
        file_exists = os.path.isfile(self.log_path)

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Only write the header for a brand-new file; resuming a run
                # must not insert a second header mid-data.
                #writer.writerow(["step", "speed", "offroad", "pos_x", "pos_z", "action", "reason"])
                writer.writerow(["step", "speed", "offroad", "pos_x", "pos_z", "action", "reason", "reward"])

    def _on_step(self) -> bool:
        """Buffers telemetry data from all environment instances for the current step.

        Called automatically by SB3 after **every** environment step.
        ``self.locals`` is a dict injected by the training algorithm and
        contains (among other keys):

        * ``"infos"``   – list of info dicts, one per parallel environment.
          Each dict is expected to carry a ``"telemetry"`` sub-dict with keys
          ``speed``, ``offroad``, ``pos_x``, ``pos_z``, and ``action``, plus
          an optional ``"terminal_reason"`` string set on episode end.
        * ``"rewards"`` – list of scalar rewards, one per parallel environment,
          for the **current** step (before any discounting).

        After appending all environment rows, the method checks whether the
        buffer has reached :attr:`flush_freq` and triggers a disk write if so.

        Returns:
            bool: Always ``True``.  Returning ``False`` from this callback
            would signal SB3 to abort training early; we never want that here.
        """
        # Iterate over every parallel environment's output for this timestep.
        for i in range(len(self.locals["infos"])):
            info = self.locals["infos"][i]
            reward = self.locals["rewards"][i]  # Capture current step reward
            tel = info["telemetry"]  # Unpack the nested telemetry sub-dict.
            self.buffer.append([
                self.num_timesteps,   # Global step counter maintained by SB3.
                tel["speed"],
                tel["offroad"],
                tel["pos_x"],
                tel["pos_z"],
                tel["action"],
                info.get("terminal_reason", ""),  # Empty string if mid-episode.
                reward,
            ])

        # Trigger a batch write once the buffer is large enough to amortise
        # the per-call file-open / flush overhead across many rows.
        if len(self.buffer) >= self.flush_freq:
            self._flush_buffer()

        return True  # Returning False would halt training; always return True.

    def _flush_buffer(self) -> None:
        """Writes all buffered telemetry rows to the CSV file in one batch.

        Using ``writer.writerows()`` (plural) with the entire buffer in a
        single call is significantly faster than opening the file and calling
        ``writerow()`` once per step, because it minimises the number of
        ``write()`` syscalls and eliminates repeated file-handle acquisition.
        After writing, the buffer is cleared to reclaim memory.
        """
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Batch write: one syscall for all buffered rows instead of N.
            writer.writerows(self.buffer)

        # Clear the buffer so memory is not held indefinitely between flushes.
        self.buffer = []

    def _on_training_end(self) -> None:
        """Flushes any remaining buffered rows when training stops.

        Because :meth:`_on_step` only flushes when the buffer reaches
        :attr:`flush_freq`, the final partial batch would otherwise be lost
        when the process exits.  This method ensures every captured row is
        persisted regardless of whether a clean flush boundary was reached
        at the end of training.  It is called by SB3 after the final
        environment step and before the callback is torn down.
        """
        self._flush_buffer()