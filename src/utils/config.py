"""
Configuration constants for the Mario Kart DS RL Agent.
Contains directory paths, RL hyperparameters, and memory addresses for the US ROM.

Note:
    ROM_PATH is resolved lazily on first access so that importing this module
    never raises — analysis scripts and utilities can import config freely on
    machines that have no ROM installed.

    All speed and off-road values stored in RAM use a **12-bit fixed-point**
    representation.  Divide the raw integer by 4096 (``value >> 12``) to obtain
    the value in natural units (e.g. km/h or a 0-1 fraction).
"""

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project Root Discovery
# ---------------------------------------------------------------------------
# Resolves to the repository root regardless of where Python is invoked from.
# __file__ → src/utils/config.py  →  .parent×3  → repo root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Save-State Path  (always available — no ROM required)
# ---------------------------------------------------------------------------
SAVE_FILE_NAME = str(ROOT_DIR / "mkds_boot.dst")  # DeSmuME save-state used to boot into race

# ---------------------------------------------------------------------------
# ROM Path  (lazy — resolved on first access via module __getattr__)
# ---------------------------------------------------------------------------
# Access as: config.ROM_PATH   (same public API as a plain constant)
_rom_path_cache: Optional[str] = None  # None until first access; cached afterwards


def _resolve_rom_path() -> str:
    """Locate the Mario Kart DS ROM inside the project's ``rom/`` directory.

    Globs ``<repo_root>/rom/*.nds`` and returns the absolute path of the first
    match.  Using a glob rather than a fixed filename lets users keep any
    regional filename without renaming.

    Returns:
        str: Absolute path to the first ``.nds`` file found in ``rom/``.

    Raises:
        FileNotFoundError: If ``rom/`` contains no ``.nds`` file, with an
            actionable message telling the user where to place the ROM.
    """
    rom_folder = ROOT_DIR / "rom"  # expected location: <repo_root>/rom/
    rom_files = list(rom_folder.glob("*.nds"))  # accept any .nds filename
    if not rom_files:
        raise FileNotFoundError(
            f"No .nds ROM found in {rom_folder}. "
            "Place your Mario Kart DS (USA) ROM in the rom/ directory."
        )
    return str(rom_files[0])


def __getattr__(name: str) -> str:
    """Module-level lazy attribute hook (PEP 562, Python 3.7+).

    Intercepts attribute lookups on this module so that ``ROM_PATH`` behaves
    like a normal module-level constant while still being computed only on
    first access.

    Args:
        name (str): The attribute name being looked up on the module.

    Returns:
        str: The resolved ROM path when ``name == "ROM_PATH"``.

    Raises:
        AttributeError: For any ``name`` other than ``"ROM_PATH"``, mimicking
            the default module attribute-error behaviour.
    """
    if name == "ROM_PATH":
        global _rom_path_cache
        if _rom_path_cache is None:
            _rom_path_cache = _resolve_rom_path()  # expensive glob — only runs once
        return _rom_path_cache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------------------------------------------------------------------
# RL Hyperparameters
# ---------------------------------------------------------------------------

# Observation image dimensions — 84×84 matches the Atari DQN benchmark so
# pre-trained CNN weights and published baselines are directly comparable.
STATE_W, STATE_H = 84, 84

# Number of consecutive frames stacked into one observation so the agent can
# infer velocity and direction (a single frame is Markovian for position only).
STACK_SIZE = 4

# Parallel DeSmuME instances used for environment stepping.
NUM_OF_INSTANCES = 4

# Action space size.  The commented-out 6-action variant includes explicit
# drift inputs; the active 3-action set keeps the policy simpler during
# initial training (gas is implicit — no brake action).
#ACTION_SPACE = 6  # 0:Straight, 1:Left, 2:Right, 3:Drift+Straight, 4:Drift+Left, 5:Drift+Right
ACTION_SPACE = 3  # 0:Straight (Gas), 1:Left (Gas+Left), 2:Right (Gas+Right)

# Discount factor — 0.99 gives roughly a 100-step effective horizon, suitable
# for a race track where rewards are sparse across a full lap.
GAMMA = 0.99

# Adam learning rate from the original Nature DQN paper (Mnih et al., 2015).
LEARNING_RATE = 0.00025

# Replay buffer capacity in transitions.  50 000 keeps memory footprint low
# while still providing sufficient diversity for stable Q-learning updates.
MEMORY_SIZE = 50000

# Mini-batch size for each gradient update.
# 128 > original 32 to better utilise modern GPU throughput.
# BATCH_SIZE = 32
BATCH_SIZE = 128

# ε-greedy exploration schedule (kept for reference / future annealing).
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 100000  # Frames over which ε decays from START → END

# ---------------------------------------------------------------------------
# Memory Pointers — US (NTSC) ROM Version
# ---------------------------------------------------------------------------
# These are *pointer* addresses: the 32-bit word at each address holds the
# base address of the relevant data structure in ARM9 RAM.

ADDR_TIMER_POINTER     = 0x0217AA34  # → race timer struct (tracks lap/split times)
ADDR_BASE_POINTER      = 0x0217ACF8  # → per-kart physics/state struct for player 1
ADDR_RACE_INFO_POINTER = 0x021755FC  # → global race-info struct (lap count, position, etc.)

# ---------------------------------------------------------------------------
# Struct Field Offsets — relative to the address stored at ADDR_BASE_POINTER
# ---------------------------------------------------------------------------
# Offsets actively used by the current agent implementation:

OFFSET_SPEED      = 0x2A8  # s32, 12-bit fixed-point → divide by 4096 for km/h equivalent
OFFSET_ANGLE      = 0x236  # s16, heading angle of the kart in world space (arbitrary units)
OFFSET_CHECKPOINT = 0x46   # u8,  last checkpoint index crossed (used to detect progress)
OFFSET_LAP        = 0x38   # u8,  current lap number (0-indexed)
OFFSET_OFFROAD    = 0xDC   # s32, 12-bit fixed-point off-road friction factor; 0 = on-road

# Offsets retained for future work (not read by the current reward/obs pipeline):

OFFSET_POS_X       = 0x80   # f32, world-space X position of the kart (metres)
OFFSET_POS_Y       = 0x84   # f32, world-space Y position (vertical axis, metres)
OFFSET_POS_Z       = 0x88   # f32, world-space Z position of the kart (metres)
OFFSET_VEL_X       = 0xA4   # f32, world-space velocity component along X axis
OFFSET_VEL_Y       = 0xA8   # f32, world-space velocity component along Y axis (vertical)
OFFSET_VEL_Z       = 0xAC   # f32, world-space velocity component along Z axis
OFFSET_MOV_DIR_X   = 0x68   # f32, X component of the kart's normalised movement direction
OFFSET_INPUT_DIR_X = 0x50   # f32, X component of the player's requested input direction
OFFSET_NORM_X      = 0x244  # f32, X component of the surface normal beneath the kart
OFFSET_NORM_Y      = 0x248  # f32, Y component of the surface normal (points up on flat road)
OFFSET_NORM_Z      = 0x24C  # f32, Z component of the surface normal beneath the kart
OFFSET_YSPEED      = 0x260  # s32, 12-bit fixed-point vertical speed (positive = ascending)
OFFSET_MAX_SPEED   = 0xD0   # s32, 12-bit fixed-point current speed cap for this kart/item state
OFFSET_TURN_LOSS   = 0x2D4  # s32, 12-bit fixed-point speed penalty applied during turning
OFFSET_AIR_SPEED   = 0x3F8  # s32, 12-bit fixed-point airborne horizontal speed
OFFSET_WALL_MULT   = 0x38C  # s32, 12-bit fixed-point speed multiplier applied on wall contact
OFFSET_PITCH       = 0x234  # s16, pitch angle of the kart (nose-up / nose-down tilt)
OFFSET_DRIFT_ANGLE = 0x388  # s16, internal drift angle accumulated during a drift manoeuvre
OFFSET_MT_CHARGE   = 0x30C  # u16, mini-turbo charge level (0 → full charge threshold)
OFFSET_MT_BOOST    = 0x23C  # u8,  frames remaining on an active mini-turbo boost
OFFSET_BOOST_ALL   = 0x238  # u8,  frames remaining on any boost (item or mini-turbo)
OFFSET_GRIP        = 0x240  # s32, 12-bit fixed-point traction/grip coefficient (1.0 = full grip)
OFFSET_AIR_FRAMES  = 0x380  # u16, consecutive frames the kart has been airborne
OFFSET_SPAWN_ID    = 0x3C4  # u8,  respawn point index used when the kart falls off-track
OFFSET_STATUS_FLAGS= 0x44   # u32, bitmask of kart status flags (airborne, boosting, drifting…)
OFFSET_PLAYER_IDX  = 0x74   # u8,  player index (0 = human / agent, 1-7 = CPU opponents)
