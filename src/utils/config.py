import os
import glob
from pathlib import Path

# Resolve the project root: config.py is in root/src/utils/
# .parent (utils) -> .parent (src) -> .parent (root)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# --- ROM & SAVE STATE PATH ---
# Locate any .nds file within root/rom/
rom_folder = ROOT_DIR / "rom"
rom_files = list(rom_folder.glob("*.nds"))

if not rom_files:
    raise FileNotFoundError(f"No .nds ROM found in {rom_folder}")

ROM_PATH = str(rom_files[0])
SAVE_FILE_NAME = str(ROOT_DIR / "mkds_boot.dst")

# --- RL Hyperparameters ---
STATE_W, STATE_H = 84, 84
STACK_SIZE = 4
#simpler input for better training.
#ACTION_SPACE = 6  # 0:Straight, 1:Left, 2:Right, 3:Drift+Straight, 4:Drift+Left, 5:Drift+Right
ACTION_SPACE = 3  # 0:Straight (Gas), 1:Left (Gas+Left), 2:Right (Gas+Right)
GAMMA = 0.99
LEARNING_RATE = 0.00025
MEMORY_SIZE = 50000
# BATCH_SIZE = 32
BATCH_SIZE = 128
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 100000  # Frames to reach EPSILON_END

# --- Pointers (US Version) ---
ADDR_TIMER_POINTER = 0x0217AA34  
ADDR_BASE_POINTER = 0x0217ACF8
ADDR_RACE_INFO_POINTER = 0x021755FC
OFFSET_SPEED = 0x2A8
OFFSET_ANGLE = 0x236
OFFSET_CHECKPOINT = 0x46
OFFSET_LAP = 0x38        
OFFSET_OFFROAD = 0xDC   