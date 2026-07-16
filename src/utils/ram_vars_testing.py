"""
RAM Variable Testing Suite for Mario Kart DS (US Version).

Used to verify memory offsets and validate telemetry extraction
before integration into the Gymnasium environment.  Launches the
DeSmuME emulator, loads the ROM and an optional save-state, then
renders a live ANSI dashboard of physics, surface, item, and race
data at roughly 60 fps.
"""

import time
import os
import math
import keyboard
from desmume.emulator import DeSmuME
from desmume.controls import keymask, Keys
import config

# --- CONFIGURATION ---
ROM_PATH = config.ROM_PATH          # Absolute path to the MKDS ROM file
SAVE_FILE_NAME = config.SAVE_FILE_NAME  # Save-state file loaded on startup if present

# --- POINTERS (US Version) ---
# These are global pointer addresses whose *values* must be dereferenced
# at runtime to reach the actual data structures in DS RAM.
ADDR_BASE_POINTER      = 0x0217ACF8  # → kart physics/state struct for local player
ADDR_RACE_INFO_POINTER = 0x021755FC  # → per-player race progress struct
ADDR_TIMER_POINTER     = 0x0217AA34  # → race timer base (frame counter at +4)
ADDR_ITEM_INFO_POINTER = 0x020FA8A4  # → item slot array base (0x210 bytes per player)
ADDR_COURSE_ID         = 0x023CDCD8  # Direct u8 — no pointer dereference needed

# --- OFFSETS (relative to dereferenced base_ptr) ---
# Position in DS fixed-point (divide by 4096 for metres)
OFFSET_POS_X       = 0x80
OFFSET_POS_Y       = 0x84
OFFSET_POS_Z       = 0x88

# Velocity components (fixed-point, same scale as position)
OFFSET_VEL_X       = 0xA4
OFFSET_VEL_Y       = 0xA8
OFFSET_VEL_Z       = 0xAC

# Movement / input direction unit vectors
OFFSET_MOV_DIR_X   = 0x68
OFFSET_INPUT_DIR_X = 0x50

# Surface normal vector (used to detect slope / wall contact)
OFFSET_NORM_X      = 0x244
OFFSET_NORM_Y      = 0x248
OFFSET_NORM_Z      = 0x24C

# Speed values (fixed-point)
OFFSET_SPEED       = 0x2A8  # Forward scalar speed along kart heading
OFFSET_YSPEED      = 0x260  # Vertical component of speed
OFFSET_MAX_SPEED   = 0xD0   # Current stat-based maximum speed cap

# Surface interaction modifiers (fixed-point multipliers around 1.0)
OFFSET_OFFROAD     = 0xDC   # Speed reduction multiplier when off-road (<1.0 = slowed)
OFFSET_TURN_LOSS   = 0x2D4  # Speed loss coefficient while turning
OFFSET_AIR_SPEED   = 0x3F8  # Speed scalar applied while airborne
OFFSET_WALL_MULT   = 0x38C  # Speed multiplier applied on wall collision

# Heading / orientation (signed 16-bit angles, full circle = 0x10000)
OFFSET_ANGLE       = 0x236  # Yaw — facing direction of the kart
OFFSET_PITCH       = 0x234  # Pitch — nose-up / nose-down tilt
OFFSET_DRIFT_ANGLE = 0x388  # Additional yaw offset accumulated during a drift

# Boost / mini-turbo state
OFFSET_MT_CHARGE   = 0x30C  # Mini-turbo charge accumulator (threshold triggers boost)
OFFSET_MT_BOOST    = 0x23C  # Remaining frames of active mini-turbo boost
OFFSET_BOOST_ALL   = 0x238  # Combined boost timer (any source)
OFFSET_GRIP        = 0x240  # Traction multiplier (1.0 = normal, <1 = slippery)
OFFSET_AIR_FRAMES  = 0x380  # Consecutive airborne frame counter
OFFSET_SPAWN_ID    = 0x3C4  # Last respawn point index
OFFSET_STATUS_FLAGS= 0x44   # Bitfield of kart state (grounded, boosting, drifting...)

# Race progress (relative to race_info_ptr)
OFFSET_PLAYER_IDX  = 0x74   # Index of this kart in the global player array (0-7)
OFFSET_LAP         = 0x38   # Current lap number (0-based)
OFFSET_CHECKPOINT  = 0x46   # Last passed checkpoint index on the course

# ---------------------------------------------------------------------------
# HELPERS — raw memory readers
# All reads use little-endian byte order, matching the ARM9 (NDS) architecture.
# ---------------------------------------------------------------------------

def read_u8(emu, addr):
    """Read one unsigned byte from emulator RAM.

    Args:
        emu: Active ``DeSmuME`` emulator instance.
        addr (int): Absolute 32-bit ARM9 RAM address to read.

    Returns:
        int: Unsigned 8-bit value (0-255).

    Note:
        NDS ARM9 memory is little-endian; single-byte reads are unambiguous.
    """
    return emu.memory.unsigned[addr]


def read_u16(emu, addr):
    """Read two bytes from emulator RAM as an unsigned 16-bit integer.

    Args:
        emu: Active ``DeSmuME`` emulator instance.
        addr (int): Absolute 32-bit ARM9 RAM address of the low byte.

    Returns:
        int: Unsigned 16-bit value (0-65535), decoded little-endian.

    Note:
        NDS ARM9 is little-endian; the byte at ``addr`` is the LSB.
    """
    return int.from_bytes(emu.memory.unsigned[addr:addr+2], 'little')


def read_s16(emu, addr):
    """Read two bytes from emulator RAM as a signed 16-bit integer.

    Args:
        emu: Active ``DeSmuME`` emulator instance.
        addr (int): Absolute 32-bit ARM9 RAM address of the low byte.

    Returns:
        int: Signed 16-bit value (-32768 to 32767), decoded little-endian.

    Note:
        Used primarily for angular values where the full circle maps to
        the range 0x0000-0xFFFF, with values above 0x7FFF being negative
        (i.e. the kart is facing left of north).
    """
    return int.from_bytes(emu.memory.unsigned[addr:addr+2], 'little', signed=True)


def read_u32(emu, addr):
    """Read four bytes from emulator RAM as an unsigned 32-bit integer.

    Args:
        emu: Active ``DeSmuME`` emulator instance.
        addr (int): Absolute 32-bit ARM9 RAM address of the low byte.

    Returns:
        int: Unsigned 32-bit value (0-4294967295), decoded little-endian.

    Note:
        Primarily used to dereference pointer addresses; a return value of
        ``0`` indicates a null pointer (data not yet initialised by the game).
    """
    return int.from_bytes(emu.memory.unsigned[addr:addr+4], 'little')


def read_s32(emu, addr):
    """Read four bytes from emulator RAM as a signed 32-bit integer.

    Args:
        emu: Active ``DeSmuME`` emulator instance.
        addr (int): Absolute 32-bit ARM9 RAM address of the low byte.

    Returns:
        int: Signed 32-bit value, decoded little-endian.

    Note:
        Most MKDS physics quantities (position, velocity, speed, grip...)
        are stored as signed 32-bit fixed-point values with 12 fractional
        bits.  Call :func:`fixed_to_float` to convert to floating-point.
    """
    return int.from_bytes(emu.memory.unsigned[addr:addr+4], 'little', signed=True)


def read_vector(emu, base_addr, offset_x):
    """Read three consecutive signed 32-bit values as an (X, Y, Z) vector.

    The three components are laid out sequentially in memory:
    ``offset_x``, ``offset_x + 4``, ``offset_x + 8``.

    Args:
        emu: Active ``DeSmuME`` emulator instance.
        base_addr (int): Dereferenced pointer to the kart struct base.
        offset_x (int): Byte offset of the X component within the struct.

    Returns:
        tuple[int, int, int]: Raw fixed-point (X, Y, Z) signed 32-bit values.
            Divide each component by 4096 to obtain real-world units.

    Note:
        MKDS stores position, velocity, and surface-normal vectors in this
        packed 3xS32 layout.  All three reads share the same ``base_addr``
        so that a single function covers all vector types.
    """
    x = read_s32(emu, base_addr + offset_x)
    y = read_s32(emu, base_addr + offset_x + 4)   # +4 bytes for Y component
    z = read_s32(emu, base_addr + offset_x + 8)   # +8 bytes for Z component
    return x, y, z


def fixed_to_float(val):
    """Convert a MKDS 12-bit fixed-point integer to a Python float.

    Args:
        val (int): Raw signed or unsigned fixed-point integer from DS RAM.

    Returns:
        float: Floating-point representation of ``val``.

    Note:
        Mario Kart DS uses a 20.12 fixed-point format throughout its physics
        engine (1 unit = 4096 raw counts).  Dividing by 4096 (2^12) recovers
        the real-world value in metres or metres/frame as appropriate.
    """
    return val / 4096.0  # 2^12 = 4096 fractional bits


#dummy version
def get_course_name(course_id):
    """Return a human-readable course name for a raw course ID byte.

    Args:
        course_id (int): Raw 8-bit course identifier read from
            ``ADDR_COURSE_ID`` (0x023CDCD8).

    Returns:
        str: Course name string, or ``'ID_<course_id>'`` for unknown IDs.

    Note:
        This is a partial lookup table covering nitro and retro cups plus
        the debug "Test Circle" track (ID 0).  IDs not listed here are
        returned in the fallback format so that new/unknown values are still
        visible in the dashboard without raising an exception.
    """
    courses = {
        0: "Test Circle",
        # --- Nitro Cup courses (IDs 1-16) ---
        1: "Figure-8 Circuit", 2: "Yoshi Falls", 3: "Cheep Cheep Beach", 4: "Luigi's Mansion",
        5: "Desert Hills", 6: "Delfino Square", 7: "Waluigi Pinball", 8: "Shroom Ridge",
        9: "DK Pass", 10: "Tick-Tock Clock", 11: "Mario Circuit", 12: "Airship Fortress",
        13: "Wario Stadium", 14: "Peach Gardens", 15: "Bowser Castle", 16: "Rainbow Road",
        # --- Retro Cup courses (IDs 32-35) ---
        32: "SNES Mario Circuit 1", 33: "N64 Moo Moo Farm", 34: "GBA Peach Circuit", 35: "GCN Luigi Circuit",
    }
    return courses.get(course_id, f"ID_{course_id}")


def get_item_name(item_id):
    """Return a human-readable item name for a raw item ID byte.

    Args:
        item_id (int): Raw 8-bit item identifier read from the player item
            slot at offset ``0x30`` within the per-player item struct.

    Returns:
        str: Item name string, or ``'Unk(<item_id>)'`` for unrecognised IDs.
            IDs 16 and 255 both represent an empty item slot (``'None'``).
    """
    items = {
        0: "Green Shell", 1: "Red Shell", 2: "Banana", 3: "Mushroom",
        4: "Star", 5: "Blue Shell", 6: "Lightning", 7: "Fake Box",
        8: "Mega Mush", 9: "Bomb", 10: "Blooper", 11: "Boo",
        12: "Gold Mush", 13: "Bullet Bill",
        16: "None",   # empty slot sentinel
        255: "None"   # uninitialized / no item
    }
    return items.get(item_id, f"Unk({item_id})")


def process_custom_controls(emu):
    """Map keyboard state to DS button inputs for manual kart control.

    Polls the keyboard every frame and adds or removes the corresponding
    DS key-mask bits so the emulator sees clean edge-free button states.

    Mapping:
        - ``W``     -> A button (accelerate)
        - ``S``     -> B button (brake / reverse)
        - ``A``     -> D-pad LEFT (steer left)
        - ``D``     -> D-pad RIGHT (steer right)
        - ``Space`` -> R shoulder (drift / hop)
        - ``Shift``  -> L shoulder (use item)

    Args:
        emu: Active ``DeSmuME`` emulator instance whose ``input`` interface
            is updated in-place.
    """
    # Keyboard processing (same as before)
    # Pre-compute bitmasks once to avoid repeated calls inside each branch
    mask_a     = keymask(Keys.KEY_A)
    mask_b     = keymask(Keys.KEY_B)
    mask_left  = keymask(Keys.KEY_LEFT)
    mask_right = keymask(Keys.KEY_RIGHT)
    mask_r     = keymask(Keys.KEY_R)
    mask_l     = keymask(Keys.KEY_L)

    # Each key is checked every frame: add_key / rm_key keeps the emulator
    # state correct regardless of whether the key was already held.
    if keyboard.is_pressed('w'): emu.input.keypad_add_key(mask_a)
    else: emu.input.keypad_rm_key(mask_a)
    if keyboard.is_pressed('s'): emu.input.keypad_add_key(mask_b)
    else: emu.input.keypad_rm_key(mask_b)
    if keyboard.is_pressed('a'): emu.input.keypad_add_key(mask_left)
    else: emu.input.keypad_rm_key(mask_left)
    if keyboard.is_pressed('d'): emu.input.keypad_add_key(mask_right)
    else: emu.input.keypad_rm_key(mask_right)
    if keyboard.is_pressed('space'): emu.input.keypad_add_key(mask_r)
    else: emu.input.keypad_rm_key(mask_r)
    if keyboard.is_pressed('shift'): emu.input.keypad_add_key(mask_l)
    else: emu.input.keypad_rm_key(mask_l)


def main():
    """Run the interactive RAM telemetry dashboard loop.

    Initialises the DeSmuME emulator, loads the MKDS ROM, and optionally
    restores a save-state so the game is immediately in a race.  Each
    iteration of the main loop:

    1. Advances the emulator by one cycle (one NDS frame ~16.7 ms).
    2. Reads all relevant RAM addresses via the pointer chain.
    3. Derives human-readable values (fixed-point conversion, surface
       classification, 2-D speed from velocity components).
    4. Renders a seven-line ANSI dashboard using cursor-home (``\\033[H``)
       and line-erase (``\\033[K``) escape codes so the terminal updates
       in-place without scrolling.
    5. Sleeps for the remainder of the 16 ms (~60 fps) budget to avoid
       spinning the CPU.

    The loop exits cleanly when the SDL window is closed.

    Note:
        Surface status is classified from the ``grip`` and ``offroad``
        fixed-point multipliers derived from TAS reverse-engineering scripts:

        - grip < 0.9  -> "SLIPPERY (Grass/Dirt)"
        - grip > 1.1  -> "STICKY"
        - offroad < 0.9 -> appends "[SLOWED]"
    """
    os.system("") # Init ANSI — emit an empty system call to enable Windows VT100 support
    emu = DeSmuME()
    try:
        emu.open(ROM_PATH)
        window = emu.create_sdl_window()
    except Exception as e:
        print(f"Error loading ROM: {e}")
        return

    # Restore a save-state if one exists so we can skip straight into a race
    if os.path.exists(SAVE_FILE_NAME):
        try: emu.savestate.load_file(SAVE_FILE_NAME)
        except: pass  # If the save-state is corrupt/incompatible, start from scratch

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Initializing...")
    time.sleep(1)  # Brief pause so the emulator can finish its own init

    while not window.has_quit():
        start_time = time.time()  # Frame start — used for the 60 fps sleep at the end
        window.process_input()          # Pump SDL events (keyboard/mouse/quit)
        emu.input.keypad_update(0)      # Reset all DS buttons to released each frame
        process_custom_controls(emu)    # Re-apply currently held keyboard keys
        emu.cycle()                     # Step the NDS CPU/GPU by one frame
        window.draw()                   # Blit the NDS framebuffer to the SDL window

        # --- MEMORY READ ---
        # Dereference the top-level pointers first; 0 means not yet initialised.
        base_ptr = read_u32(emu, ADDR_BASE_POINTER)           # -> kart physics struct
        race_info_ptr = read_u32(emu, ADDR_RACE_INFO_POINTER) # -> race progress struct
        timer_base = read_u32(emu, ADDR_TIMER_POINTER)        # -> timer struct
        item_base = read_u32(emu, ADDR_ITEM_INFO_POINTER)     # -> item slot array
        course_id = read_u8(emu, ADDR_COURSE_ID)              # Direct byte, no deref
        course_name = get_course_name(course_id)

        if base_ptr != 0:
            #physics/surface
            # --- POSITION, VELOCITY, SURFACE NORMAL (raw fixed-point) ---
            raw_x, raw_y, raw_z = read_vector(emu, base_ptr, OFFSET_POS_X)
            vx, vy, vz = read_vector(emu, base_ptr, OFFSET_VEL_X)
            nx, ny, nz = read_vector(emu, base_ptr, OFFSET_NORM_X)

            # Scalar speed values
            speed_val = read_s32(emu, base_ptr + OFFSET_SPEED)    # Forward speed
            yspeed_val = read_s32(emu, base_ptr + OFFSET_YSPEED)  # Vertical speed
            max_spd = read_s32(emu, base_ptr + OFFSET_MAX_SPEED)

            # --- SURFACE VARIABLES ---
            offroad_mod = read_s32(emu, base_ptr + OFFSET_OFFROAD)  # Speed multiplier off-road
            grip_val = read_s32(emu, base_ptr + OFFSET_GRIP)        # Traction multiplier

            # Heading angles (signed 16-bit; full rotation = 0x10000 counts)
            angle = read_s16(emu, base_ptr + OFFSET_ANGLE)
            drift_angle = read_s16(emu, base_ptr + OFFSET_DRIFT_ANGLE)

            # Mini-turbo charge counter and combined status bitfield
            mt_charge = read_s32(emu, base_ptr + OFFSET_MT_CHARGE)
            flags = read_u32(emu, base_ptr + OFFSET_STATUS_FLAGS)

            # --- RACE PROGRESS ---
            lap = -1
            checkpoint = -1
            if race_info_ptr != 0:
                lap = read_u8(emu, race_info_ptr + OFFSET_LAP)
                checkpoint = read_u8(emu, race_info_ptr + OFFSET_CHECKPOINT)

            # --- GLOBAL TIMER ---
            # The frame counter lives at timer_base + 4 (4-byte little-endian int)
            global_timer = 0
            if timer_base != 0:
                global_timer = int.from_bytes(emu.memory.unsigned[timer_base+4:timer_base+8], 'little')

            # --- ITEM SLOT ---
            item_name = "None"
            player_idx = read_u8(emu, base_ptr + OFFSET_PLAYER_IDX)
            if item_base != 0:
                # Each player occupies 0x210 bytes in the item array
                player_item_addr = item_base + (0x210 * player_idx)
                raw_item_id = read_u8(emu, player_item_addr + 0x30)  # Item type byte
                item_count = read_u8(emu, player_item_addr + 0x38)   # Quantity held
                if item_count > 0: item_name = get_item_name(raw_item_id)

            #calculations from TAS scripts
            # True 2-D ground speed: magnitude of the horizontal velocity vector
            real_speed = math.sqrt(vx**2 + vz**2) / 4096.0  # Y excluded (vertical component)
            grip_float = fixed_to_float(grip_val)
            offroad_float = fixed_to_float(offroad_mod)

            # Classify the surface the kart is currently on
            surface_status = "ROAD"
            if grip_float < 0.9: surface_status = "SLIPPERY (Grass/Dirt)"  # Grip below normal
            elif grip_float > 1.1: surface_status = "STICKY"               # Unusual high-grip surface

            if offroad_float < 0.9: surface_status += " [SLOWED]"  # Off-road speed penalty active

            # --- DASHBOARD ---
            # \033[H  -> move cursor to top-left (home position)
            # \033[K  -> erase from cursor to end of line (prevents stale chars)
            l0 = f"COURSE: {course_name} (ID: {course_id})"
            l1 = f"POS: X:{fixed_to_float(raw_x):7.2f} Y:{fixed_to_float(raw_y):7.2f} Z:{fixed_to_float(raw_z):7.2f}"
            l2 = f"SPD: Fwd:{fixed_to_float(speed_val):6.2f} Max:{fixed_to_float(max_spd):6.2f} Real2D:{real_speed:6.2f}"
            l3 = f"SURF: Grip:{grip_float:4.2f} Offrd:{offroad_float:4.2f} -> {surface_status}"
            l4 = f"ANG: Fac:{angle:6} Drft:{drift_angle:5} | Flags:{flags:08X}"
            l5 = f"BST: MT:{mt_charge:4} | Lap:{lap} CP:{checkpoint} | Time:{global_timer}"
            l6 = f"ITM: {item_name}"

            output = (
                f"\033[H"
                f"{l0}\033[K\n"
                f"{l1}\033[K\n"
                f"{l2}\033[K\n"
                f"{l3}\033[K\n"
                f"{l4}\033[K\n"
                f"{l5}\033[K\n"
                f"{l6}\033[K"
            )
            print(output, flush=True)  # flush=True ensures the terminal updates immediately

        else:
            # Pointer not yet valid — game is still in menus or loading
            print(f"\033[HWaiting for race start... {time.time():.1f}\033[K", end='\r')

        # --- 60 fps FRAME CAP ---
        elapsed = time.time() - start_time
        if elapsed < 0.016: time.sleep(0.016 - elapsed)  # Sleep for the remainder of the ~16 ms budget

if __name__ == "__main__":
    main()