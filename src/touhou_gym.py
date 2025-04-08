import sys
import logging
import gc
from os.path import pathsep, abspath

import gymnasium
import tinyscaler
from PIL import Image
from gymnasium import spaces
import numpy as np
import random

from pytouhou.game import NextStage, GameOver
from pytouhou.ui.gamerunner import GameRunner
from pytouhou.utils.random import Random
from pytouhou.games.eosd.game import Game, Common
from pytouhou.games.eosd.interface import Interface
from pytouhou.lib.sdl import show_simple_message_box
from pytouhou.resource.loader import Loader
from pytouhou.ui.opengl import backend
from pytouhou.ui.window import Window

from gym_utils import get_entities, get_boss, bullet_intersects_hitbox, closest_point

offset_x = 32
offset_y = 16
x = 384
y = 448

UP = 16
DOWN = 32
LEFT = 64
RIGHT = 128
SHOOT = 1
FOCUS = 4

actions = [
    SHOOT,
    FOCUS,
    SHOOT | FOCUS,
    UP,
    UP | LEFT,
    UP | LEFT | SHOOT,
    UP | LEFT | FOCUS,
    UP | LEFT | SHOOT | FOCUS,
    UP | RIGHT,
    UP | RIGHT | SHOOT,
    UP | RIGHT | FOCUS,
    UP | RIGHT | SHOOT | FOCUS,
    DOWN,
    DOWN | LEFT,
    DOWN | LEFT | SHOOT,
    DOWN | LEFT | FOCUS,
    DOWN | LEFT | SHOOT | FOCUS,
    DOWN | RIGHT,
    DOWN | RIGHT | SHOOT,
    DOWN | RIGHT | FOCUS,
    DOWN | RIGHT | SHOOT | FOCUS,
    LEFT,
    LEFT | SHOOT,
    LEFT | FOCUS,
    LEFT | SHOOT | FOCUS,
    RIGHT,
    RIGHT | SHOOT,
    RIGHT | FOCUS,
    RIGHT | SHOOT | FOCUS
]

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
            fps_limit=-1,
            unlock_fps=True,
            disable_render=False,
    ):
        self.gl_options = {
            'flavor': 'compatibility',
            'version': 2.1,
            'double-buffer': None,
            'frontend': 'glfw',
            'backend': 'opengl'
        }
        self.disable_render = disable_render
        self.render_mode = 'rgb_array'
        self.resource_path = abspath(game_path)
        self.fps_limit = fps_limit
        self.unlock_fps = unlock_fps

        self.observation_space = spaces.Dict({
            'game_player': spaces.Box(low=-1, high=1,shape=(5,), dtype=np.float32),
            'game_boss': spaces.Box(low=-1, high=1,shape=(2,), dtype=np.float32),
            'game_closest_bullet': spaces.Box(low=-1, high=1,shape=(2,), dtype=np.float32),
            'game_closest_item': spaces.Box(low=-1, high=1,shape=(2,), dtype=np.float32),
            'game_closest_enemy': spaces.Box(low=-1, high=1,shape=(2,), dtype=np.float32),
            'pife_player_bullets': spaces.Box(low=-1, high=1,shape=(100, 4), dtype=np.float32),
            'pife_game_bullets': spaces.Box(
                low=-1,
                high=1,
                shape=(250, 4),
                dtype=np.float32
            ),
            'pife_game_enemies': spaces.Box(
                low=-1,
                high=1,
                shape=(20, 2),
                dtype=np.float32
            ),
            'pife_game_items': spaces.Box(
                low=-1,
                high=1,
                shape=(20, 2),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Discrete(len(actions))
        self.rewards = 0
        self.current_score = 0

        self.characters = [0]
        self.continues = 0
        self.random_stage = random_stage
        self.stage_num = stage_num if not random_stage else random.randint(1, 6)
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
        framebuffer = self.renderer.get_framebuffer(Interface.width, Interface.height, greyscale=False)
        img = np.frombuffer(framebuffer, dtype=np.uint8).reshape((Interface.height, Interface.width, 4))
        return np.flipud(img)

    def _start(self):
        self.resource_loader = Loader(self.resource_path)

        try:
            self.resource_loader.scan_archives(game_data_locations)
            backend.init(self.gl_options)
        except IOError:
            show_simple_message_box(u'Some data files were not found, did you forget the -p option?')
            sys.exit(1)
        except AssertionError as e:
            logging.error(f'Backend failed to initialize: {e}')
            sys.exit(1)

        self.window = CustomWindow(backend, self.disable_render, Interface.width, Interface.height, fps_limit=self.fps_limit, frameskip=0, unlock_fps=self.unlock_fps)
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
        self.stage_num = self.stage_num if not self.random_stage else random.randint(1, 6)
        self.rank = 3
        self.difficulty = 16

        self.renderer = backend.GameRenderer(self.resource_loader, self.window)
        self.common = Common(self.resource_loader, self.characters, self.continues)
        self.interface = Interface(self.resource_loader, self.common.players[0])
        self.common.interface = self.interface
        self.runner = GameRunner(self.window, self.renderer, self.common, self.resource_loader)
        self.window.set_runner(self.runner)

        self.prng = Random(seed=seed if seed is not None else -1)
        self.game = Game(
            resource_loader=self.resource_loader,
            stage=self.stage_num,
            rank=self.rank,
            difficulty=self.difficulty,
            common=self.common,
            prng=self.prng
        )
        self.runner.load_game(self.game, self.game.background, self.game.std.bgms, None, None)
        self.game.players[0].lives = 0
        self.current_score = 0
        self.rewards = 0

    def _get_obs(self):
        px, py = self.game.players[0].x, self.game.players[0].y
        h = self.game.players[0].sht.hitbox
        player_np = np.array(
            [(px + h) / x, (px - h) / x, (py + h) / y, (py - h) / y],
            dtype=np.float32)

        def fill_array(entities, shape, transform_fn):
            arr = np.full(shape, -1, dtype=np.float32)
            valid = [(i, e) for i, e in enumerate(entities) if e]
            if valid:
                idxs, ents = zip(*valid)
                arr[list(idxs)] = np.array([transform_fn(e) for e in ents], dtype=np.float32)
            return arr

        bullets_np = fill_array(
            get_entities(self.game.bullets, m=250), (250, 4),
            lambda b: (b.x / x, b.y / y, b.dx / x, b.dy / y)
        )

        players_bullets_np = fill_array(
            get_entities(self.game.players_bullets, m=100), (100, 4),
            lambda b: (b.x / x, b.y / y, b.dx / x, b.dy / y)
        )

        enemies_np = fill_array(
            get_entities(self.game.enemies, m=20), (20, 2),
            lambda e: (e.x / x, e.y / y)
        )

        items_np = fill_array(
            get_entities(self.game.items, m=20), (20, 2),
            lambda i: (i.x / x, i.y / y)
        )

        boss_np = np.asarray(get_boss(self.game.boss), dtype=np.float32)

        closest_bullet = closest_point(bullets_np, (px, py))
        closest_item = closest_point(items_np, (px, py))
        closest_enemy = closest_point(enemies_np, (px, py))

        return {
            'game_player': player_np,
            'game_boss': boss_np,
            'game_closest_bullet': closest_bullet,
            'game_closest_item': closest_item,
            'game_closest_enemy': closest_enemy,
            'pife_game_bullets': bullets_np,
            'pife_player_bullets': players_bullets_np,
            'pife_game_enemies': enemies_np,
            'pife_game_items': items_np
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset(seed)

        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        terminated = False

        keystate = actions[action]

        self.window.set_keystate(keystate)

        try:
            self.window.run_frame()
        except (NextStage, GameOver):
            terminated = True
        observation = self._get_obs()
        is_dead = self.starting_lives > self.game.players[0].lives
        self.game.players[0].lives = 0

        reward = self.game.players[0].rewards - self.rewards
        self.rewards = self.game.players[0].rewards

        raw_bullets = get_entities(self.game.bullets, m=250)
        bullet_array = np.full((250, 4), -1, dtype=np.float32)
        for i, b in enumerate(raw_bullets):
            if b:
                bullet_array[i] = (b.x, b.y, b.dx, b.dy)

        valid_hits, _ = bullet_intersects_hitbox(
            self.game.players[0].x,
            self.game.players[0].y,
            self.game.players[0].sht.hitbox,
            bullet_array
        )

        bullet_intersect = np.any(valid_hits)

        if is_dead:
            reward -= 10
            terminated = True
        elif bullet_intersect:
            reward -= 0.01  # neg reward for being in bullet vector
        else:
            reward += 0.01 # reward for staying alive

        return observation, reward, terminated, False, {}



    def close(self):
        self.window.set_runner(None)
        gc.collect()
