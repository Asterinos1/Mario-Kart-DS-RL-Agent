import time
import os
import math
import sys
import keyboard
from desmume.emulator import DeSmuME
from desmume.controls import keymask, Keys

# --- CONFIGURATION ---
ROM_PATH = 'C:/Users/PC/Desktop/DS-DSi/Library/mkds_usa.nds'
SAVE_FILE_NAME = 'C:/Users/PC/Documents/GitHub/MKDS-RL-Agent/mkds_boot.dst'

# --- POINTERS (US Version) ---
ADDR_BASE_POINTER      = 0x0217ACF8
ADDR_RACE_INFO_POINTER = 0x021755FC
ADDR_TIMER_POINTER     = 0x0217AA34
ADDR_ITEM_INFO_POINTER = 0x020FA8A4
ADDR_COURSE_ID         = 0x023CDCD8  # <--- New Pointer for Course ID

# --- OFFSETS ---
OFFSET_POS_X       = 0x80
OFFSET_POS_Y       = 0x84
OFFSET_POS_Z       = 0x88
OFFSET_VEL_X       = 0xA4
OFFSET_VEL_Y       = 0xA8
OFFSET_VEL_Z       = 0xAC
OFFSET_MOV_DIR_X   = 0x68
OFFSET_INPUT_DIR_X = 0x50
OFFSET_NORM_X      = 0x244
OFFSET_NORM_Y      = 0x248  # Upward normal (1.0 = Flat)
OFFSET_NORM_Z      = 0x24C

OFFSET_SPEED       = 0x2A8
OFFSET_YSPEED      = 0x260
OFFSET_MAX_SPEED   = 0xD0
OFFSET_OFFROAD     = 0xDC   # <--- Offroad Speed Modifier
OFFSET_TURN_LOSS   = 0x2D4
OFFSET_AIR_SPEED   = 0x3F8
OFFSET_WALL_MULT   = 0x38C

OFFSET_ANGLE       = 0x236
OFFSET_PITCH       = 0x234
OFFSET_DRIFT_ANGLE = 0x388

OFFSET_MT_CHARGE   = 0x30C
OFFSET_MT_BOOST    = 0x23C
OFFSET_BOOST_ALL   = 0x238
OFFSET_GRIP        = 0x240  # <--- Surface Grip
OFFSET_AIR_FRAMES  = 0x380
OFFSET_SPAWN_ID    = 0x3C4
OFFSET_STATUS_FLAGS= 0x44

OFFSET_PLAYER_IDX  = 0x74
OFFSET_LAP         = 0x38
OFFSET_CHECKPOINT  = 0x46

# --- HELPERS ---
def read_u8(emu, addr): return emu.memory.unsigned[addr]
def read_u16(emu, addr): return int.from_bytes(emu.memory.unsigned[addr:addr+2], 'little')
def read_s16(emu, addr): return int.from_bytes(emu.memory.unsigned[addr:addr+2], 'little', signed=True)
def read_u32(emu, addr): return int.from_bytes(emu.memory.unsigned[addr:addr+4], 'little')
def read_s32(emu, addr): return int.from_bytes(emu.memory.unsigned[addr:addr+4], 'little', signed=True)

def read_vector(emu, base_addr, offset_x):
    x = read_s32(emu, base_addr + offset_x)
    y = read_s32(emu, base_addr + offset_x + 4)
    z = read_s32(emu, base_addr + offset_x + 8)
    return x, y, z

def fixed_to_float(val):
    return val / 4096.0

#dummy version
def get_course_name(course_id):
    courses = {
        0: "Test Circle",
        1: "Figure-8 Circuit", 2: "Yoshi Falls", 3: "Cheep Cheep Beach", 4: "Luigi's Mansion",
        5: "Desert Hills", 6: "Delfino Square", 7: "Waluigi Pinball", 8: "Shroom Ridge",
        9: "DK Pass", 10: "Tick-Tock Clock", 11: "Mario Circuit", 12: "Airship Fortress",
        13: "Wario Stadium", 14: "Peach Gardens", 15: "Bowser Castle", 16: "Rainbow Road",
        32: "SNES Mario Circuit 1", 33: "N64 Moo Moo Farm", 34: "GBA Peach Circuit", 35: "GCN Luigi Circuit",
    }
    return courses.get(course_id, f"ID_{course_id}")

def get_item_name(item_id):
    items = {
        0: "Green Shell", 1: "Red Shell", 2: "Banana", 3: "Mushroom",
        4: "Star", 5: "Blue Shell", 6: "Lightning", 7: "Fake Box",
        8: "Mega Mush", 9: "Bomb", 10: "Blooper", 11: "Boo",
        12: "Gold Mush", 13: "Bullet Bill", 16: "None", 255: "None"
    }
    return items.get(item_id, f"Unk({item_id})")

def process_custom_controls(emu):
    # Keyboard processing (same as before)
    mask_a = keymask(Keys.KEY_A)
    mask_b = keymask(Keys.KEY_B)
    mask_left = keymask(Keys.KEY_LEFT)
    mask_right = keymask(Keys.KEY_RIGHT)
    mask_r = keymask(Keys.KEY_R)
    mask_l = keymask(Keys.KEY_L)
    
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
    os.system("") # Init ANSI
    emu = DeSmuME()
    try:
        emu.open(ROM_PATH)
        window = emu.create_sdl_window()
    except Exception as e:
        print(f"Error loading ROM: {e}")
        return

    if os.path.exists(SAVE_FILE_NAME):
        try: emu.savestate.load_file(SAVE_FILE_NAME)
        except: pass

    os.system('cls' if os.name == 'nt' else 'clear')
    print("Initializing...")
    time.sleep(1)

    while not window.has_quit():
        start_time = time.time()
        window.process_input()
        emu.input.keypad_update(0)
        process_custom_controls(emu)
        emu.cycle()
        window.draw()

        # --- MEMORY READ ---
        base_ptr = read_u32(emu, ADDR_BASE_POINTER)
        race_info_ptr = read_u32(emu, ADDR_RACE_INFO_POINTER)
        timer_base = read_u32(emu, ADDR_TIMER_POINTER)
        item_base = read_u32(emu, ADDR_ITEM_INFO_POINTER)
        course_id = read_u8(emu, ADDR_COURSE_ID)
        course_name = get_course_name(course_id)

        if base_ptr != 0:
            #physics/surface
            raw_x, raw_y, raw_z = read_vector(emu, base_ptr, OFFSET_POS_X)
            vx, vy, vz = read_vector(emu, base_ptr, OFFSET_VEL_X)
            nx, ny, nz = read_vector(emu, base_ptr, OFFSET_NORM_X)
            
            speed_val = read_s32(emu, base_ptr + OFFSET_SPEED)
            yspeed_val = read_s32(emu, base_ptr + OFFSET_YSPEED)
            max_spd = read_s32(emu, base_ptr + OFFSET_MAX_SPEED)
            
            # --- SURFACE VARIABLES ---
            offroad_mod = read_s32(emu, base_ptr + OFFSET_OFFROAD)
            grip_val = read_s32(emu, base_ptr + OFFSET_GRIP)
            
            angle = read_s16(emu, base_ptr + OFFSET_ANGLE)
            drift_angle = read_s16(emu, base_ptr + OFFSET_DRIFT_ANGLE)
            mt_charge = read_s32(emu, base_ptr + OFFSET_MT_CHARGE)
            flags = read_u32(emu, base_ptr + OFFSET_STATUS_FLAGS)

            lap = -1
            checkpoint = -1
            if race_info_ptr != 0:
                lap = read_u8(emu, race_info_ptr + OFFSET_LAP)
                checkpoint = read_u8(emu, race_info_ptr + OFFSET_CHECKPOINT)
            
            global_timer = 0
            if timer_base != 0:
                global_timer = int.from_bytes(emu.memory.unsigned[timer_base+4:timer_base+8], 'little')

            item_name = "None"
            player_idx = read_u8(emu, base_ptr + OFFSET_PLAYER_IDX)
            if item_base != 0:
                player_item_addr = item_base + (0x210 * player_idx)
                raw_item_id = read_u8(emu, player_item_addr + 0x30)
                item_count = read_u8(emu, player_item_addr + 0x38)
                if item_count > 0: item_name = get_item_name(raw_item_id)

            #calculations from TAS scripts
            real_speed = math.sqrt(vx**2 + vz**2) / 4096.0
            grip_float = fixed_to_float(grip_val)
            offroad_float = fixed_to_float(offroad_mod)

            surface_status = "ROAD"
            if grip_float < 0.9: surface_status = "SLIPPERY (Grass/Dirt)"
            elif grip_float > 1.1: surface_status = "STICKY"
            
            if offroad_float < 0.9: surface_status += " [SLOWED]"

            # --- DASHBOARD ---
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
            print(output, flush=True)

        else:
            print(f"\033[HWaiting for race start... {time.time():.1f}\033[K", end='\r')

        elapsed = time.time() - start_time
        if elapsed < 0.016: time.sleep(0.016 - elapsed)

if __name__ == "__main__":
    main()