import numpy as np
import cv2
import os
import math
from desmume.emulator import DeSmuME, SCREEN_WIDTH, SCREEN_HEIGHT_BOTH
from desmume.controls import keymask, Keys
from PIL import Image
from src.utils import config

class MKDSEnvLegacy:
    def __init__(self):
        # Initialize emulator
        self.emu = DeSmuME()
        self.emu.open(config.ROM_PATH)
        self.window = self.emu.create_sdl_window()

        # Action mapping
        self.action_map = self._setup_actions()
        
        # Circuit Data for Reward Logic
        self.chkpnt_ang = [
            16380, 16380, 20805, 24812, 28650, -30861, -23773, -17975, -15386, -11925,
            -8233, -8233, -8233, -8233, -8233, -8233, -14276, -18273, -27352, 29514, 
            26495, 18165, 16380, 16380, 16380, 16380
        ]

        self.prev_checkpoint = 0
        self.prev_lap = 0

    def _setup_actions(self):
        ACCEL, LEFT, RIGHT, DRIFT = keymask(Keys.KEY_A), keymask(Keys.KEY_LEFT), keymask(Keys.KEY_RIGHT), keymask(Keys.KEY_R)
        return {
            0: [ACCEL], 1: [ACCEL, LEFT], 2: [ACCEL, RIGHT],
            3: [ACCEL, DRIFT], 4: [ACCEL, DRIFT, LEFT], 5: [ACCEL, DRIFT, RIGHT]
        }

    def _get_state(self):
        """Original PIL-based capture for legacy compatibility."""
        img = self.emu.screenshot()
        top_screen = img.crop((0, 0, 256, 192))
        gray = top_screen.resize((config.STATE_W, config.STATE_H), Image.Resampling.LANCZOS).convert('L')
        return np.array(gray, dtype=np.uint8)

    def _read_ram(self):
        mem = self.emu.memory.unsigned
        base_ptr = int.from_bytes(mem[config.ADDR_BASE_POINTER:config.ADDR_BASE_POINTER+4], 'little')
        race_ptr = int.from_bytes(mem[config.ADDR_RACE_INFO_POINTER:config.ADDR_RACE_INFO_POINTER+4], 'little')
        
        if base_ptr == 0 or race_ptr == 0:
            return 0.0, 0, 0, 0, 1.0

        speed = int.from_bytes(mem[base_ptr + config.OFFSET_SPEED:base_ptr + config.OFFSET_SPEED + 4], 'little', signed=True) / 4096.0
        angle = int.from_bytes(mem[base_ptr + config.OFFSET_ANGLE:base_ptr + config.OFFSET_ANGLE + 2], 'little', signed=True)
        checkpoint = mem[race_ptr + config.OFFSET_CHECKPOINT]
        lap = mem[race_ptr + config.OFFSET_LAP]
        offroad = int.from_bytes(mem[base_ptr + config.OFFSET_OFFROAD:base_ptr + config.OFFSET_OFFROAD + 4], 'little', signed=True) / 4096.0
        return speed, angle, checkpoint, lap, offroad

    def calculate_reward(self, speed, angle, checkpoint, lap, offroad_mod):
        reward = speed * 2.0
        if checkpoint > self.prev_checkpoint: reward += 15.0
        if lap > self.prev_lap: reward += 100.0
        if offroad_mod < 0.9: reward *= 0.5
        
        self.prev_checkpoint = checkpoint
        self.prev_lap = lap
        return reward

    def step(self, action_idx):
        self.emu.input.keypad_update(0)
        for key in self.action_map[action_idx]:
            self.emu.input.keypad_add_key(key)
        
        for _ in range(4): self.emu.cycle() 
        self.window.draw() 

        next_raw_frame = self._get_state()
        speed, angle, cp, lap, offroad = self._read_ram()
        reward = self.calculate_reward(speed, angle, cp, lap, offroad)
        
        done = True if lap >= 3 else False
        return next_raw_frame, reward, done

    def reset(self):
        if os.path.exists(config.SAVE_FILE_NAME):
            self.emu.savestate.load_file(config.SAVE_FILE_NAME) 
        self.prev_checkpoint = 0
        self.prev_lap = 0
        return self._get_state()