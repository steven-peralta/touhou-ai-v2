import sys
import logging
import gc
from os.path import pathsep, abspath

import gymnasium
from gymnasium import spaces
import numpy as np
import random

from pytouhou.game import NextStage, GameOver
from pytouhou.ui.gamerunner import GameRunner
from pytouhou.utils.random import Random
from pytouhou.games.eosd.game import Game, Common
from pytouhou.games.eosd.interface import Interface
from pytouhou.game.music import MusicPlayer
from pytouhou.lib.sdl import show_simple_message_box
from pytouhou.resource.loader import Loader
from pytouhou.ui.opengl import backend
from pytouhou.ui.window import Window

from gym_utils import get_entities, get_boss, bullet_intersects_hitbox, closest_point, GAME_WIDTH, GAME_HEIGHT

UP = 16
DOWN = 32
LEFT = 64
RIGHT = 128
SHOOT = 1
BOMB = 2
FOCUS = 4

DIRECTIONS = [0, UP, DOWN, LEFT, RIGHT, UP | LEFT, UP | RIGHT, DOWN | LEFT, DOWN | RIGHT]

game_data_locations = (pathsep.join(('CM.DAT', 'th06*_CM.DAT', '*CM.DAT', '*cm.dat')),
                       pathsep.join(('ST.DAT', 'th6*ST.DAT', '*ST.DAT', '*st.dat')),
                       pathsep.join(('IN.DAT', 'th6*IN.DAT', '*IN.DAT', '*in.dat')),
                       pathsep.join(('MD.DAT', 'th6*MD.DAT', '*MD.DAT', '*md.dat')),
                       pathsep.join(('102h.exe', '102*.exe', '東方紅魔郷.exe', '*.exe')))

class CustomWindow(Window):
    def __init__(self, backend, disable_render, width, height, fps_limit, frameskip, unlock_fps):
        super().__init__(backend=backend, disable_render=disable_render, width=width, height=height, fps_limit=fps_limit, frameskip=frameskip, no_delay=unlock_fps)
        self.keystate = 0

    def set_keystate(self, keystate):
        self.keystate = keystate

    def get_keystate(self):
        return self.keystate



class TouhouGym(gymnasium.Env):

    def __init__(
            self,
            game_path='./res/game/',
            stage_num=1,
            random_stage=False,
            stages=None,
            fps_limit=-1,
            unlock_fps=True,
            disable_render=False,
            mortal=False,
    ):
        self.gl_options = {
            'flavor': 'compatibility',
            'version': 2.1,
            'double-buffer': None,
            'frontend': 'glfw',
            'backend': 'opengl'
        }
        self.disable_render = disable_render
        self.mortal = mortal
        self.render_mode = 'rgb_array'
        self.resource_path = abspath(game_path)
        self.fps_limit = fps_limit
        self.unlock_fps = unlock_fps

        self.observation_space = spaces.Dict({
            'game_player': spaces.Box(low=-1, high=1,shape=(4,), dtype=np.float32),
            'game_stage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'game_boss': spaces.Box(low=-1, high=1,shape=(2,), dtype=np.float32),
            'game_closest_bullet': spaces.Box(low=-1, high=1,shape=(4,), dtype=np.float32),
            'game_closest_item': spaces.Box(low=-1, high=1,shape=(4,), dtype=np.float32),
            'game_closest_enemy': spaces.Box(low=-1, high=1,shape=(4,), dtype=np.float32),
            'pife_game_bullets': spaces.Box(
                low=-1,
                high=1,
                shape=(640, 5),
                dtype=np.float32
            ),
            'pife_game_enemies': spaces.Box(
                low=-1,
                high=1,
                shape=(20, 3),
                dtype=np.float32
            ),
            'pife_game_items': spaces.Box(
                low=-1,
                high=1,
                shape=(20, 3),
                dtype=np.float32
            )
        })

        # [direction(9), shoot(2), focus(2), bomb(2)]
        self.action_space = spaces.MultiDiscrete([9, 2, 2, 2])
        self.current_score = 0
        self.last_graze = 0
        self.still_frames = 0
        self.last_px = 0.0
        self.last_py = 0.0

        self.characters = [0]
        self.continues = 0
        self.random_stage = random_stage
        self.stages = stages
        if stages:
            self.stage_num = random.choice(stages)
        elif random_stage:
            self.stage_num = random.randint(1, 6)
        else:
            self.stage_num = stage_num
        self.rank = 3
        self.difficulty = 16

        self.resource_loader = None
        self.game = None
        self.prng = None
        self.runner = None
        self.interface = None
        self.common = None
        self.renderer = None
        self.window = None

        self.starting_lives = 0

        self._start()

    def render(self):
        if self.disable_render:
            return None
        framebuffer = self.renderer.get_framebuffer(Interface.width, Interface.height, greyscale=False)
        img = np.frombuffer(framebuffer, dtype=np.uint8).reshape((Interface.height, Interface.width, 4))
        return np.flipud(img)

    def _start(self):
        self.resource_loader = Loader(self.resource_path)

        try:
            self.resource_loader.scan_archives(game_data_locations)
        except IOError:
            show_simple_message_box(u'Some data files were not found, did you forget the -p option?')
            sys.exit(1)

        if not self.disable_render:
            try:
                backend.init(self.gl_options)
            except AssertionError as e:
                logging.error(f'Backend failed to initialize: {e}')
                sys.exit(1)

            self.window = CustomWindow(backend, False, Interface.width, Interface.height, fps_limit=self.fps_limit, frameskip=0, unlock_fps=self.unlock_fps)
            self.renderer = backend.GameRenderer(self.resource_loader, self.window)
            self.common = Common(self.resource_loader, self.characters, self.continues)
            self.interface = Interface(self.resource_loader, self.common.players[0])
            self.common.interface = self.interface
            self.runner = GameRunner(self.window, self.renderer, self.common, self.resource_loader)
            self.window.set_runner(self.runner)

    def _reset(self, seed=-1):
        if self.game is not None:
            self.game.cleanup()

        self.characters = [0]
        self.continues = 0
        if self.stages:
            self.stage_num = random.choice(self.stages)
        elif self.random_stage:
            self.stage_num = random.randint(1, 6)

        self.rank = 3
        self.difficulty = 16

        self.common = Common(self.resource_loader, self.characters, self.continues)
        self.interface = Interface(self.resource_loader, self.common.players[0])
        self.common.interface = self.interface

        self.prng = Random(seed=seed if seed is not None else -1)
        self.game = Game(
            resource_loader=self.resource_loader,
            stage=self.stage_num,
            rank=self.rank,
            difficulty=self.difficulty,
            common=self.common,
            prng=self.prng
        )

        null_player = MusicPlayer()
        self.game.music = null_player
        self.game.sfx_player = null_player

        if not self.disable_render:
            self.renderer = backend.GameRenderer(self.resource_loader, self.window)
            self.runner = GameRunner(self.window, self.renderer, self.common, self.resource_loader)
            self.window.set_runner(self.runner)
            self.runner.load_game(self.game, self.game.background, self.game.std.bgms, None, None)

        # Mortal mode: 0 lives (die on first hit). Invincible mode: many lives.
        self.game.players[0].lives = 0 if self.mortal else 9999

        # Fast-forward past the empty pre-spawn phase
        while len(self.game.enemies) == 0 and len(self.game.bullets) == 0:
            self.game.run_iter([0])

        self.current_score = self.game.players[0].score
        self.last_graze = self.game.players[0].graze
        self.still_frames = 0
        self.last_px = self.game.players[0].x
        self.last_py = self.game.players[0].y

    def _get_obs(self):
        px, py = self.game.players[0].x, self.game.players[0].y
        h = self.game.players[0].sht.hitbox
        player_np = np.array(
            [(px + h) / GAME_WIDTH, (px - h) / GAME_WIDTH,
             (py + h) / GAME_HEIGHT, (py - h) / GAME_HEIGHT],
            dtype=np.float32)

        def fill_array(entities, shape, transform_fn):
            arr = np.full(shape, -1, dtype=np.float32)
            valid = [(i, e) for i, e in enumerate(entities) if e]
            if valid:
                idxs, ents = zip(*valid)
                arr[list(idxs)] = np.array([transform_fn(e) for e in ents], dtype=np.float32)
            return arr

        bullets_np = fill_array(
            get_entities(self.game.bullets, m=640), (640, 5),
            lambda b: (b.x / GAME_WIDTH, b.y / GAME_HEIGHT,
                       b.dx / GAME_WIDTH, b.dy / GAME_HEIGHT,
                       b._bullet_type.type_id / 9.0)
        )

        enemies_np = fill_array(
            get_entities(self.game.enemies, m=20), (20, 3),
            lambda e: (e.x / GAME_WIDTH, e.y / GAME_HEIGHT,
                       (e._type & 0xFF) / 255.0)
        )

        items_np = fill_array(
            get_entities(self.game.items, m=20), (20, 3),
            lambda i: (i.x / GAME_WIDTH, i.y / GAME_HEIGHT,
                       i._type / 6.0)
        )

        boss_np = np.asarray(get_boss(self.game.boss), dtype=np.float32)

        # Use normalized player coords to match normalized entity arrays
        norm_px, norm_py = px / GAME_WIDTH, py / GAME_HEIGHT
        closest_bullet = closest_point(bullets_np, (norm_px, norm_py))
        closest_item = closest_point(items_np, (norm_px, norm_py))
        closest_enemy = closest_point(enemies_np, (norm_px, norm_py))

        stage_np = np.array([self.stage_num / 6.0], dtype=np.float32)

        return {
            'game_player': player_np,
            'game_stage': stage_np,
            'game_boss': boss_np,
            'game_closest_bullet': closest_bullet,
            'game_closest_item': closest_item,
            'game_closest_enemy': closest_enemy,
            'pife_game_bullets': bullets_np,
            'pife_game_enemies': enemies_np,
            'pife_game_items': items_np
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset(seed)

        observation = self._get_obs()

        return observation, {}

    def _get_raw_bullet_array(self):
        raw_bullets = get_entities(self.game.bullets, m=640)
        bullet_array = np.full((640, 4), -1, dtype=np.float32)
        for i, b in enumerate(raw_bullets):
            if b:
                bullet_array[i] = (b.x, b.y, b.dx, b.dy)
        return bullet_array

    def step(self, action):
        terminated = False

        direction, shoot, focus, bomb = action
        keystate = DIRECTIONS[direction]
        if shoot: keystate |= SHOOT
        if focus: keystate |= FOCUS
        if bomb:  keystate |= BOMB

        if self.disable_render:
            try:
                self.game.run_iter([keystate])
            except (NextStage, GameOver):
                terminated = True
        else:
            self.window.set_keystate(keystate)
            try:
                self.window.run_frame()
            except (NextStage, GameOver):
                terminated = True

        # Skip cutscenes/dialogue by fast-forwarding with shoot pressed
        while self.game.msg_wait and not terminated:
            try:
                if self.disable_render:
                    self.game.run_iter([SHOOT])
                else:
                    self.window.set_keystate(SHOOT)
                    self.window.run_frame()
            except (NextStage, GameOver):
                terminated = True
            except AttributeError:
                # Engine can crash accessing boss_callback during dialogue transitions
                break

        observation = self._get_obs()

        # Detect hit via lives dropping
        expected_lives = 0 if self.mortal else 9999
        was_hit = self.game.players[0].lives < expected_lives
        self.game.players[0].lives = expected_lives

        # Normalized score delta as base reward, with graze contribution removed
        score_delta = self.game.players[0].score - self.current_score
        graze_delta = self.game.players[0].graze - self.last_graze
        self.current_score = self.game.players[0].score
        self.last_graze = self.game.players[0].graze

        real_score_delta = score_delta - (graze_delta * 500)
        reward = real_score_delta / 1000.0

        # Gradient bullet danger
        bullet_array = self._get_raw_bullet_array()
        valid_hits, dists = bullet_intersects_hitbox(
            self.game.players[0].x,
            self.game.players[0].y,
            self.game.players[0].sht.hitbox,
            bullet_array
        )

        px, py = self.game.players[0].x, self.game.players[0].y

        # Hit handling: mortal mode terminates, invincible mode penalizes
        if was_hit:
            if self.mortal:
                reward -= 5.0
                terminated = True
            else:
                reward -= 2.0

        # Danger penalty
        if np.any(valid_hits):
            threatening_dists = dists[valid_hits]
            danger_score = np.sum(1.0 / (threatening_dists + 1.0))
            reward -= 0.1 * min(danger_score, 5.0)

        # Stillness penalty: penalize staying in the same position for >60 frames
        moved = abs(px - self.last_px) > 1.0 or abs(py - self.last_py) > 1.0
        if moved:
            self.still_frames = 0
        else:
            self.still_frames += 1
        if self.still_frames > 60:
            reward -= 0.05

        # Edge penalty: penalize top, left, and right edges (bottom is okay)
        edge_margin = 32.0
        if py < edge_margin:  # too close to top
            reward -= 0.02
        if px < edge_margin:  # too close to left
            reward -= 0.02
        if px > GAME_WIDTH - edge_margin:  # too close to right
            reward -= 0.02

        self.last_px = px
        self.last_py = py

        return observation, reward, terminated, False, {}



    def close(self):
        if self.window is not None:
            self.window.set_runner(None)
        gc.collect()
