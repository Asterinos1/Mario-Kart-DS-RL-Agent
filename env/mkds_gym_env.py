"""Mario Kart DS Gymnasium Environment.

This module defines ``MKDSEnv``, a custom ``gymnasium.Env`` that wraps
the DeSmuME NDS emulator so that reinforcement-learning agents can train
on Mario Kart DS.  Observations are grayscale frames captured from the
top screen.  Rewards are shaped with a set of watchdog checks (backward
driving, timeout, collision, stuck detection) plus a per-step speed bonus
and checkpoint bonuses.  All game state (speed, position, lap progress) is
read directly from NDS RAM via DeSmuME's memory interface.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
import math
from desmume.emulator import DeSmuME, SCREEN_WIDTH, SCREEN_HEIGHT_BOTH
from src.utils import config

class MKDSEnv(gym.Env):
    """Gymnasium environment for Mario Kart DS.

    Uses DeSmuME for emulation and memory access for reward shaping.
    Observations are single-channel (grayscale) crops of the top screen,
    downscaled to ``(config.STATE_H, config.STATE_W, 1)`` for efficiency.

    Attributes:
        emu (DeSmuME): The running DeSmuME emulator instance.
        window: SDL render window used for live visualisation, or ``None``
            when ``visualize=False``.
        action_space (spaces.Discrete): Discrete action space whose size is
            defined by ``config.ACTION_SPACE``.
        observation_space (spaces.Box): Box observation space with shape
            ``(config.STATE_H, config.STATE_W, 1)`` and dtype ``uint8``.
        action_map (dict[int, list[int]]): Mapping from discrete action
            index to a list of DeSmuME keymask values to press simultaneously.
        prev_checkpoint (int): Checkpoint index reached in the previous step,
            used to detect backward driving and award checkpoint bonuses.
        prev_lap (int): Lap count from the previous step.
        prev_speed (float): Kart speed (fixed-point units) from the previous
            step, used to detect sudden collision-induced speed drops.
        stuck_counter (int): Number of consecutive steps where the kart has
            moved less than 50 world units; triggers the stuck watchdog.
        last_pos (tuple[int, int, int]): World-space ``(X, Y, Z)`` position
            from the previous step, used for stuck detection.
        last_cp_time_stamp (int): Internal race-timer value (60 ticks/s) at
            the last checkpoint advance; used for the timeout watchdog.
    """

    def __init__(self, visualize=False):
        """Initialises the emulator, spaces, and internal tracking state.

        Args:
            visualize (bool): When ``True``, creates an SDL window so the
                emulator renders frames in real time.  Defaults to ``False``
                for headless training.
        """
        super(MKDSEnv, self).__init__()
        self.emu = DeSmuME()
        self.emu.open(config.ROM_PATH)
        self.window = None
        if visualize:
            self.window = self.emu.create_sdl_window()

        # Discrete action space — size controlled by config so experiments
        # can toggle expanded action sets without touching this file.
        self.action_space = spaces.Discrete(config.ACTION_SPACE)

        # Single-channel (grayscale) observation to reduce CNN input size.
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(config.STATE_H, config.STATE_W, 1), 
                                            dtype=np.uint8)

        self.action_map = self._setup_actions()
        
        # Tracking variables for Watchdogs
        self.prev_checkpoint = 0       # Last known checkpoint index
        self.prev_lap = 0              # Last known lap number
        self.prev_speed = 0.0          # Last known speed (fixed-point scaled)
        self.stuck_counter = 0         # Consecutive low-movement steps
        self.last_pos = (0, 0, 0)      # Last known world-space position
        self.last_cp_time_stamp = 0    # Track internal time of last CP change

    def _setup_actions(self):
        """Maps discrete actions to DeSmuME keymasks.

        Builds a dictionary that translates each integer action index into a
        list of DeSmuME keymask values that should be held simultaneously
        during that step.  The commented-out block shows an extended 6-action
        set that includes drift; only the 3-action set is active.

        Returns:
            dict[int, list[int]]: Mapping of the form::

                {
                    0: [ACCEL],          # Accelerate straight
                    1: [ACCEL, LEFT],    # Accelerate + steer left
                    2: [ACCEL, RIGHT],   # Accelerate + steer right
                }
        """
        from desmume.controls import keymask, Keys
        ACCEL, LEFT, RIGHT = keymask(Keys.KEY_A), keymask(Keys.KEY_LEFT), keymask(Keys.KEY_RIGHT)
        if config.ACTION_SPACE == 6:
            DRIFT = keymask(Keys.KEY_R)
            return {
                0: [ACCEL], 
                1: [ACCEL, LEFT], 
                2: [ACCEL, RIGHT],
                3: [ACCEL, DRIFT], 
                4: [ACCEL, DRIFT, LEFT], 
                5: [ACCEL, DRIFT, RIGHT]
            }
        return {
            0: [ACCEL],         # Straight
            1: [ACCEL, LEFT],   # Left
            2: [ACCEL, RIGHT]   # Right
        }

    def _get_obs(self):
        """Captures the top screen and processes it for the CNN.

        Reads the full dual-screen RGBX framebuffer from the emulator, crops
        the top 192 rows (top DS screen), converts to grayscale, and resizes
        to ``(config.STATE_W, config.STATE_H)`` before adding a channel dim.

        Returns:
            np.ndarray: Processed observation of shape
                ``(config.STATE_H, config.STATE_W, 1)`` with dtype ``uint8``.
                Pixel values range ``[0, 255]``.
        """
        raw_mv = self.emu.display_buffer_as_rgbx()
        # Full dual-screen buffer: height = SCREEN_HEIGHT_BOTH (384), width = SCREEN_WIDTH (256), 4 channels (RGBX)
        img = np.frombuffer(raw_mv, dtype=np.uint8).reshape(SCREEN_HEIGHT_BOTH, SCREEN_WIDTH, 4)
        
        # [cite_start]Crop Top Screen (First 192 pixels) [cite: 7]
        # The NDS top screen occupies rows 0-191; the bottom touch screen is rows 192-383.
        top_screen = img[:192, :, :3]  # Drop the unused X (padding) channel
        
        # Resize and Grayscale for CNN efficiency
        gray = cv2.cvtColor(top_screen, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (config.STATE_W, config.STATE_H), interpolation=cv2.INTER_AREA)
        # Add channel dimension so shape is (H, W, 1) to match observation_space
        return np.expand_dims(resized, axis=-1)

    def _read_race_time(self):
        """Reads the internal 32-bit race timer (60 ticks per second).

        Follows a two-level pointer: first reads a pointer address from
        ``config.ADDR_TIMER_POINTER``, then dereferences that pointer to
        obtain the 32-bit tick count.

        Returns:
            int: Current race timer value in emulator ticks (60 ticks = 1 s).
                Returns ``0`` if the pointer is null (race not yet started).

        Note:
            The timer runs at 60 ticks per second, matching the NDS display
            refresh rate.  All timeout durations in ``step()`` are expressed
            in these ticks.
        """
        mem = self.emu.memory.unsigned
        # Read the pointer, then the value it points to
        ptr_val = int.from_bytes(mem[config.ADDR_TIMER_POINTER : config.ADDR_TIMER_POINTER + 4], 'little')
        if ptr_val == 0:  # Null pointer - race data not yet loaded
            return 0
        return int.from_bytes(mem[ptr_val : ptr_val + 4], 'little')

    def _read_ram(self):
        """Reads physics and race progress from NDS RAM.

        Follows two base pointers - one for the kart physics struct and one
        for the race-info struct - then reads offsets defined in ``config``.
        All multi-byte values are little-endian.  Speed and offroad values
        are stored as 20.12 fixed-point integers and are scaled by dividing
        by 4096 to yield floating-point units.

        Returns:
            tuple: A 6-element tuple ``(speed, angle, checkpoint, lap,
                offroad, pos)`` where:

                * **speed** (*float*): Kart forward speed in scaled world
                  units (fixed-point value / 4096).  Negative when reversing.
                * **angle** (*int*): Signed 16-bit heading angle in raw NDS
                  angular units.
                * **checkpoint** (*int*): Index of the last checkpoint gate
                  the kart passed through (unsigned byte).
                * **lap** (*int*): Current lap number (unsigned byte).
                * **offroad** (*float*): Offroad/surface-friction factor
                  (fixed-point / 4096); values below 1.0 indicate grass or
                  other slow surfaces.
                * **pos** (*tuple[int, int, int]*): World-space position
                  ``(X, Y, Z)`` as signed 32-bit integers in NDS world units.

            If either base pointer is null (data not yet loaded), returns the
            safe fallback ``(0.0, 0, 0, 0, 1.0, (0, 0, 0))``.

        Note:
            Fixed-point scaling: NDS physics values use a 20.12 fixed-point
            representation, so dividing by ``4096`` (2^12) converts them to
            human-readable floating-point units.
        """
        mem = self.emu.memory.unsigned
    
        base_ptr = int.from_bytes(mem[config.ADDR_BASE_POINTER:config.ADDR_BASE_POINTER+4], 'little')
        race_ptr = int.from_bytes(mem[config.ADDR_RACE_INFO_POINTER:config.ADDR_RACE_INFO_POINTER+4], 'little')
        
        if base_ptr == 0 or race_ptr == 0:  # Guard: pointers valid only mid-race
            return 0.0, 0, 0, 0, 1.0, (0,0,0)

        # Dynamic values with scaling
        # Divide by 4096 (2^12) to convert 20.12 fixed-point to float
        speed = int.from_bytes(mem[base_ptr + config.OFFSET_SPEED:base_ptr + config.OFFSET_SPEED + 4], 'little', signed=True) / 4096.0
        angle = int.from_bytes(mem[base_ptr + config.OFFSET_ANGLE:base_ptr + config.OFFSET_ANGLE + 2], 'little', signed=True)
        checkpoint = mem[race_ptr + config.OFFSET_CHECKPOINT]  # Unsigned byte; wraps at track end
        lap = mem[race_ptr + config.OFFSET_LAP]                # Unsigned byte lap counter
        # offroad < 1.0 means grass/dirt; exactly 1.0 means normal tarmac
        offroad = int.from_bytes(mem[base_ptr + config.OFFSET_OFFROAD:base_ptr + config.OFFSET_OFFROAD + 4], 'little', signed=True) / 4096.0
        
        # World-space position: three consecutive signed 32-bit words at offsets
        # 0x80 (X), 0x84 (Y - vertical axis in NDS space), 0x88 (Z)
        pos = (
            int.from_bytes(mem[base_ptr + 0x80 : base_ptr + 0x84], 'little', signed=True),  # X: lateral
            int.from_bytes(mem[base_ptr + 0x84 : base_ptr + 0x88], 'little', signed=True),  # Y: vertical (height)
            int.from_bytes(mem[base_ptr + 0x88 : base_ptr + 0x8C], 'little', signed=True)   # Z: forward/depth
        )
        return speed, angle, checkpoint, lap, offroad, pos

    def step(self, action):
        """Executes one environment step (4 emulator cycles).

        Applies the selected action for 4 emulator cycles (~1/15 s at 60 fps),
        reads the resulting game state, evaluates all watchdog termination
        conditions in priority order, and computes the shaped reward.

        Args:
            action (int): Integer index into ``self.action_map``.  Must be in
                ``[0, config.ACTION_SPACE)``.

        Returns:
            tuple: A 5-element tuple ``(obs, reward, terminated, truncated,
                info)`` where:

                * **obs** (*np.ndarray*): Grayscale top-screen observation of
                  shape ``(config.STATE_H, config.STATE_W, 1)``, dtype
                  ``uint8``.
                * **reward** (*float*): Shaped scalar reward for this step.
                  Negative for terminal failure states, positive for progress.
                * **terminated** (*bool*): ``True`` when a terminal condition
                  fires (backward driving, collision, stuck, finish, or
                  severe timeout).
                * **truncated** (*bool*): ``True`` when a non-fatal time limit
                  is reached (currently also sets ``terminated``).
                * **info** (*dict*): Auxiliary diagnostic data with keys:

                    - ``"telemetry"`` (*dict*): ``speed``, ``offroad``,
                      ``pos_x``, ``pos_y``, ``pos_z``, ``action``.
                    - ``"terminal_reason"`` (*str | None*): Human-readable
                      label for the termination cause, or ``None`` if the
                      episode is still running.
        """
        # --- APPLY ACTION ---
        # Clear all keys first to avoid sticky inputs from the previous step
        self.emu.input.keypad_update(0)
        for key in self.action_map[action]:
            self.emu.input.keypad_add_key(key)
        # Step emulator and update window
        for _ in range(4):  # 4 cycles = 1 rendered frame at full 60 fps
            self.emu.cycle()   
        if self.window is not None:
            self.window.draw()

        obs = self._get_obs()
        speed, angle, cp, lap, offroad, pos = self._read_ram()
        current_time = self._read_race_time()
        
        terminated = False
        truncated = False
        reward = 0.0
        reason = "driving"

        # --- WATCHDOG LOGIC ---
        # Watchdogs are evaluated in priority order; the first that fires wins.
        
        # 1. Backward Driving Detection
        # If the checkpoint index decreased without crossing the finish line the
        # kart is driving the wrong way - terminate immediately with a large penalty.
        # We guard against lap rollover by checking if the drop is small (less than 10).
        if cp < self.prev_checkpoint and lap == self.prev_lap:
            if self.prev_checkpoint - cp < 10:
                terminated = True
                reward = -50.0
                reason = "backward"
        
        # 2. CP Change Timeout (Internal Timer Logic)
        # 300 internal ticks = 5 seconds
        # Reset the timestamp whenever progress is made (new CP or new lap).
        if cp > self.prev_checkpoint or lap > self.prev_lap:
            self.last_cp_time_stamp = current_time  # Progress made - reset clock
        elif (current_time - self.last_cp_time_stamp) > 300:
            # No checkpoint for >300 ticks (~5 s); effectively a hard time-out
            truncated = True
            terminated = True
            reward = -15.0
            reason = "timeout"


        # 3. Sudden Collision / Wall Scraping (Speed drop > 50%)
        # A large negative delta combined with near-zero speed is a strong
        # signal of a wall impact rather than an intentional slow-down.
        speed_drop = self.prev_speed - speed  # Positive when speed decreased
        if speed < 5 and speed_drop > 3:
            terminated = True
            reward = -30.0
            reason = "collision"


        # 4. Aggressive Stuck Detection
        # Euclidean distance in 3-D world space between current and previous position.
        dist = math.dist(pos, self.last_pos)
        if dist < 50:  # Less than 50 world units - consider the kart stationary
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0  # Meaningful movement - reset counter
        
        if self.stuck_counter > 80:  # ~5-6 seconds of no meaningful progress
            terminated = True
            reward = -20.0
            reason = "stuck"
        
        # --- STANDARD REWARD (If not terminated/truncated) ---
        if not (terminated or truncated):
            # Base reward proportional to speed encourages the agent to go fast
            reward = speed * 2.0
            if cp > self.prev_checkpoint:  # Bonus for reaching a new checkpoint gate
                reward += 15.0
            if offroad < 0.9:  # Penalize grass - halve the reward when off-track
                reward *= 0.5
            
            # Lap 4 is the finish condition (laps are 1-indexed internally)
            if lap > 3: 
                terminated = True
                reward += 100.0  # Large completion bonus
                reason = "finished"


        # Update historical trackers for use in the next step's watchdog checks
        self.prev_checkpoint, self.prev_lap = cp, lap
        self.last_pos, self.prev_speed = pos, speed
        
        info = {
            "telemetry": {
                "speed": speed,
                "offroad": offroad,
                "pos_x": pos[0],
                "pos_y": pos[1],  # Vertical in DS space
                "pos_z": pos[2],
                "action": action
            },
            "terminal_reason": reason if terminated else None
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to the boot save state.

        Loads the pre-saved emulator state from ``config.SAVE_FILE_NAME``
        (which should place the game at the race start line) and zeroes all
        watchdog tracking variables so the new episode starts cleanly.

        Args:
            seed (int | None): Optional RNG seed forwarded to the parent
                ``gymnasium.Env.reset()`` for reproducibility.  Mario Kart DS
                itself is deterministic given the same inputs, so this mainly
                affects any stochastic wrappers.
            options (dict | None): Not currently used; reserved for future
                configuration overrides (e.g. track selection).

        Returns:
            tuple: A 2-element tuple ``(obs, info)`` where:

                * **obs** (*np.ndarray*): Initial grayscale observation of
                  shape ``(config.STATE_H, config.STATE_W, 1)``, dtype
                  ``uint8``, captured immediately after the save state loads.
                * **info** (*dict*): Empty dict ``{}``; included to satisfy
                  the Gymnasium API contract.
        """
        super().reset(seed=seed)
        # Reload the boot save state instead of closing/opening
        # This is much faster than a full emulator restart and avoids the
        # race-select menus that would otherwise need to be navigated.
        if os.path.exists(config.SAVE_FILE_NAME):
            self.emu.savestate.load_file(config.SAVE_FILE_NAME) 
        
        # Reset counters and timers
        self.stuck_counter = 0
        self.prev_speed = 0.0
        self.prev_checkpoint = 0
        self.prev_lap = 0
        # Initialise the CP timestamp to the current race time so the timeout
        # watchdog does not fire immediately on the first step.
        self.last_cp_time_stamp = self._read_race_time()
        
        return self._get_obs(), {}

    def close(self):
        """Cleanly destroys the emulator instance to release memory and resources."""
        if hasattr(self, 'emu') and self.emu is not None:
            self.emu.destroy()
