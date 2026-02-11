from pathlib import Path

"""
Configuration constants for the Mario Kart DS RL Agent.
Contains directory paths, RL hyperparameters, and memory addresses for the US ROM.
"""

# Project Root Discovery
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
NUM_OF_INSTANCES = 4
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
# --- Base pointers ---
ADDR_TIMER_POINTER = 0x0217AA34  
ADDR_BASE_POINTER = 0x0217ACF8
ADDR_RACE_INFO_POINTER = 0x021755FC

# --- Offsets for accesing other variables ---
# used in the current version of the project
OFFSET_SPEED = 0x2A8
OFFSET_ANGLE = 0x236
OFFSET_CHECKPOINT = 0x46
OFFSET_LAP = 0x38        
OFFSET_OFFROAD = 0xDC   

# not utilized by kept for future work
OFFSET_POS_X       = 0x80
OFFSET_POS_Y       = 0x84
OFFSET_POS_Z       = 0x88
OFFSET_VEL_X       = 0xA4
OFFSET_VEL_Y       = 0xA8
OFFSET_VEL_Z       = 0xAC
OFFSET_MOV_DIR_X   = 0x68
OFFSET_INPUT_DIR_X = 0x50
OFFSET_NORM_X      = 0x244
OFFSET_NORM_Y      = 0x248  
OFFSET_NORM_Z      = 0x24C
OFFSET_YSPEED      = 0x260
OFFSET_MAX_SPEED   = 0xD0 
OFFSET_TURN_LOSS   = 0x2D4
OFFSET_AIR_SPEED   = 0x3F8
OFFSET_WALL_MULT   = 0x38C
OFFSET_PITCH       = 0x234
OFFSET_DRIFT_ANGLE = 0x388
OFFSET_MT_CHARGE   = 0x30C
OFFSET_MT_BOOST    = 0x23C
OFFSET_BOOST_ALL   = 0x238
OFFSET_GRIP        = 0x240  
OFFSET_AIR_FRAMES  = 0x380
OFFSET_SPAWN_ID    = 0x3C4
OFFSET_STATUS_FLAGS= 0x44
OFFSET_PLAYER_IDX  = 0x74
