import os
import csv
from stable_baselines3.common.callbacks import BaseCallback

class MKDSMetricsCallback(BaseCallback):
    """
    Custom callback for logging race telemetry to CSV during training.
    """
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_path = os.path.join(log_dir, "telemetry_log.csv")
        self.buffer = []
        self.flush_freq = 5000

    def _on_training_start(self):
        """Initializes the CSV file header."""
        file_exists = os.path.isfile(self.log_path)
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "speed", "offroad", "pos_x", "pos_z", "action", "reason"])

    def _on_step(self) -> bool:
        """Buffers telemetry data from all environment instances."""
        for i in range(len(self.locals["infos"])):
            info = self.locals["infos"][i]
            tel = info["telemetry"]
            self.buffer.append([
                self.num_timesteps, tel["speed"], tel["offroad"], 
                tel["pos_x"], tel["pos_z"], tel["action"], info.get("terminal_reason", "")
            ])

        if len(self.buffer) >= self.flush_freq:
            self._flush_buffer()
        return True

    def _flush_buffer(self):
        """Writes the buffer to the CSV file."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.buffer)
        self.buffer = []

    def _on_training_end(self):
        self._flush_buffer()