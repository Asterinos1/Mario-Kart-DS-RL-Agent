import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import math
from desmume.emulator import DeSmuME, SCREEN_WIDTH, SCREEN_HEIGHT_BOTH
from src.utils import config

class MKDSEnv(gym.Env):
    def __init__(self):
        super(MKDSEnv, self).__init__()
        # Initialize emulator [cite: 112]
        self.emu = DeSmuME()
        self.emu.open(config.ROM_PATH)
        self.window = self.emu.create_sdl_window()
        
        # Action Space: 6 Discrete actions
        self.action_space = spaces.Discrete(config.ACTION_SPACE)
        
        # Observation Space: 84x84 Grayscale
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(config.STATE_H, config.STATE_W, 1), 
                                            dtype=np.uint8)

        # Action mapping
        self.action_map = self._setup_actions()
        
        # Circuit Data
        self.chkpnt_ang = [
            16380, 16380, 20805, 24812, 28650, -30861, -23773, -17975, -15386, -11925,
            -8233, -8233, -8233, -8233, -8233, -8233, -14276, -18273, -27352, 29514, 
            26495, 18165, 16380, 16380, 16380, 16380
        ]

        # Tracking variables
        self.prev_checkpoint = 0
        self.prev_lap = 0
        self.stuck_counter = 0
        self.last_pos = (0, 0, 0)

    def _setup_actions(self):
        from desmume.controls import keymask, Keys
        ACCEL, LEFT, RIGHT, DRIFT = keymask(Keys.KEY_A), keymask(Keys.KEY_LEFT), keymask(Keys.KEY_RIGHT), keymask(Keys.KEY_R)
        return {
            0: [ACCEL], 1: [ACCEL, LEFT], 2: [ACCEL, RIGHT],
            3: [ACCEL, DRIFT], 4: [ACCEL, DRIFT, LEFT], 5: [ACCEL, DRIFT, RIGHT]
        }

    def _get_obs(self):
        """Optimized Frame Capture using raw RGBX buffer[cite: 143]."""
        # Get memoryview of the RGBX buffer [cite: 143]
        raw_mv = self.emu.display_buffer_as_rgbx() 
        # Convert to numpy without copying
        img = np.frombuffer(raw_mv, dtype=np.uint8).reshape(SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4)
        # Crop Top Screen (192 lines) [cite: 7]
        top_screen = img[:192, :, :3] 
        # Resize using OpenCV (faster than PIL)
        gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (config.STATE_W, config.STATE_H), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)

    def _read_ram(self):
        # Direct unsigned access via memory property [cite: 126]
        mem = self.emu.memory.unsigned
        base_ptr = int.from_bytes(mem[config.ADDR_BASE_POINTER:config.ADDR_BASE_POINTER+4], 'little')
        race_ptr = int.from_bytes(mem[config.ADDR_RACE_INFO_POINTER:config.ADDR_RACE_INFO_POINTER+4], 'little')
        
        if base_ptr == 0 or race_ptr == 0:
            return 0.0, 0, 0, 0, 1.0, (0,0,0)

        speed = int.from_bytes(mem[base_ptr + config.OFFSET_SPEED:base_ptr + config.OFFSET_SPEED + 4], 'little', signed=True) / 4096.0
        angle = int.from_bytes(mem[base_ptr + config.OFFSET_ANGLE:base_ptr + config.OFFSET_ANGLE + 2], 'little', signed=True)
        checkpoint = mem[race_ptr + config.OFFSET_CHECKPOINT]
        lap = mem[race_ptr + config.OFFSET_LAP]
        offroad = int.from_bytes(mem[base_ptr + config.OFFSET_OFFROAD:base_ptr + config.OFFSET_OFFROAD + 4], 'little', signed=True) / 4096.0
        pos = (
            int.from_bytes(mem[base_ptr + 0x80 : base_ptr + 0x84], 'little', signed=True),
            int.from_bytes(mem[base_ptr + 0x84 : base_ptr + 0x88], 'little', signed=True),
            int.from_bytes(mem[base_ptr + 0x88 : base_ptr + 0x8C], 'little', signed=True)
        )
        return speed, angle, checkpoint, lap, offroad, pos

    def step(self, action):
        self.emu.input.keypad_update(0)
        for key in self.action_map[action]:
            self.emu.input.keypad_add_key(key)
        
        for _ in range(4): self.emu.cycle() 
        self.window.draw() 

        obs = self._get_obs()
        speed, angle, cp, lap, offroad, pos = self._read_ram()
        
        # Reward Logic
        reward = speed * 2.0
        if cp > self.prev_checkpoint: reward += 15.0
        if offroad < 0.9: reward *= 0.5
        
        # Termination Logic (Watchdogs)
        terminated = False
        truncated = False
        
        # 1. Stuck Detection
        dist = math.dist(pos, self.last_pos)
        if dist < 100: # Adjust based on 1/4096 scale
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        if self.stuck_counter > 150: # Stuck for ~10 seconds
            terminated = True
            reward -= 20.0
            
        if lap >= 3: terminated = True
        
        self.prev_checkpoint, self.prev_lap, self.last_pos = cp, lap, pos
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if os.path.exists(config.SAVE_FILE_NAME):
            self.emu.savestate.load_file(config.SAVE_FILE_NAME) 
        self.stuck_counter = 0
        return self._get_obs(), {}