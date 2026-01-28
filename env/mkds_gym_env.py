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
        self.emu = DeSmuME()
        self.emu.open(config.ROM_PATH)
        self.window = self.emu.create_sdl_window()
        
        # Action Space: 6 Discrete actions
        self.action_space = spaces.Discrete(config.ACTION_SPACE)
        
        # Observation Space: 84x84 Grayscale
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(config.STATE_H, config.STATE_W, 1), 
                                            dtype=np.uint8)

        self.action_map = self._setup_actions()
        
        # Tracking variables for Watchdogs
        self.prev_checkpoint = 0
        self.prev_lap = 0
        self.prev_speed = 0.0
        self.stuck_counter = 0
        self.last_pos = (0, 0, 0)
        self.last_cp_time_stamp = 0 # Track internal time of last CP change

    def _setup_actions(self):
        from desmume.controls import keymask, Keys
        ACCEL, LEFT, RIGHT, DRIFT = keymask(Keys.KEY_A), keymask(Keys.KEY_LEFT), keymask(Keys.KEY_RIGHT), keymask(Keys.KEY_R)
        return {
            0: [ACCEL], 
            1: [ACCEL, LEFT], 
            2: [ACCEL, RIGHT],
            3: [ACCEL, DRIFT], 
            4: [ACCEL, DRIFT, LEFT], 
            5: [ACCEL, DRIFT, RIGHT]
        }

    def _get_obs(self):
        raw_mv = self.emu.display_buffer_as_rgbx() 
        img = np.frombuffer(raw_mv, dtype=np.uint8).reshape(SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4)
        
        # [cite_start]Crop Top Screen (First 192 pixels) [cite: 7]
        top_screen = img[:192, :, :3] 
        
        # Resize and Grayscale for CNN efficiency
        gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (config.STATE_W, config.STATE_H), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)

    def _read_race_time(self):
        """Reads the internal 32-bit race timer (60 ticks per second)."""
        mem = self.emu.memory.unsigned
        # Read the pointer, then the value it points to
        ptr_val = int.from_bytes(mem[config.ADDR_TIMER_POINTER : config.ADDR_TIMER_POINTER + 4], 'little')
        if ptr_val == 0:
            return 0
        return int.from_bytes(mem[ptr_val : ptr_val + 4], 'little')

    def _read_ram(self):
        mem = self.emu.memory.unsigned
        
        # Read Pointers
        base_ptr = int.from_bytes(mem[config.ADDR_BASE_POINTER:config.ADDR_BASE_POINTER+4], 'little')
        race_ptr = int.from_bytes(mem[config.ADDR_RACE_INFO_POINTER:config.ADDR_RACE_INFO_POINTER+4], 'little')
        
        if base_ptr == 0 or race_ptr == 0:
            return 0.0, 0, 0, 0, 1.0, (0,0,0)

        # Dynamic values with scaling
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
        # [cite_start]Apply inputs [cite: 21, 22]
        self.emu.input.keypad_update(0)
        for key in self.action_map[action]:
            self.emu.input.keypad_add_key(key)
        
        # [cite_start]Step emulator and update window [cite: 138, 12]
        for _ in range(4): 
            self.emu.cycle() 
        self.window.draw() 

        obs = self._get_obs()
        speed, angle, cp, lap, offroad, pos = self._read_ram()
        current_time = self._read_race_time()
        
        terminated = False
        truncated = False
        reward = 0.0

        # --- WATCHDOG LOGIC ---
        
        # 1. Backward Driving Detection
        if cp < self.prev_checkpoint and lap == self.prev_lap:
            terminated = True
            reward = -50.0
        
        # 2. CP Change Timeout (Internal Timer Logic)
        # 300 internal ticks = 5 seconds
        if cp > self.prev_checkpoint or lap > self.prev_lap:
            self.last_cp_time_stamp = current_time
        elif (current_time - self.last_cp_time_stamp) > 10:
            truncated = True
            terminated = True
            reward = -15.0

        # 3. Sudden Collision / Wall Scraping (Speed drop > 50%)
        speed_drop = self.prev_speed - speed
        if speed < 5 and speed_drop > 3:
            terminated = True
            reward = -30.0

        # 4. Aggressive Stuck Detection
        dist = math.dist(pos, self.last_pos)
        if dist < 50: 
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        if self.stuck_counter > 80: # ~5-6 seconds of no meaningful progress
            terminated = True
            reward = -20.0

        # --- STANDARD REWARD (If not terminated/truncated) ---
        if not (terminated or truncated):
            reward = speed * 2.0
            if cp > self.prev_checkpoint: 
                reward += 15.0
            if offroad < 0.9: # Penalize grass
                reward *= 0.5
            
            if lap >= 3: 
                terminated = True
                reward += 100.0

        # Update historical trackers
        self.prev_checkpoint, self.prev_lap = cp, lap
        self.last_pos, self.prev_speed = pos, speed
        
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reload the boot save state instead of closing/opening 
        if os.path.exists(config.SAVE_FILE_NAME):
            self.emu.savestate.load_file(config.SAVE_FILE_NAME) 
        
        # Reset counters and timers
        self.stuck_counter = 0
        self.prev_speed = 0.0
        self.prev_checkpoint = 0
        self.prev_lap = 0
        self.last_cp_time_stamp = self._read_race_time()
        
        return self._get_obs(), {}