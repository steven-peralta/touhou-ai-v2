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
    UP,
    UP | LEFT,
    UP | LEFT | SHOOT,
    UP | LEFT | SHOOT | FOCUS,
    UP | RIGHT,
    UP | RIGHT | SHOOT,
    UP | RIGHT | SHOOT | FOCUS,
    DOWN,
    DOWN | LEFT,
    DOWN | LEFT | SHOOT,
    DOWN | LEFT | SHOOT | FOCUS,
    DOWN | RIGHT,
    DOWN | RIGHT | SHOOT,
    DOWN | RIGHT | SHOOT | FOCUS,
    LEFT,
    LEFT | SHOOT,
    LEFT | SHOOT | FOCUS,
    RIGHT,
    RIGHT | SHOOT,
    RIGHT | SHOOT | FOCUS
]

game_data_locations = (pathsep.join(('CM.DAT', 'th06*_CM.DAT', '*CM.DAT', '*cm.dat')),
                       pathsep.join(('ST.DAT', 'th6*ST.DAT', '*ST.DAT', '*st.dat')),
                       pathsep.join(('IN.DAT', 'th6*IN.DAT', '*IN.DAT', '*in.dat')),
                       pathsep.join(('MD.DAT', 'th6*MD.DAT', '*MD.DAT', '*md.dat')),
                       pathsep.join(('102h.exe', '102*.exe', '東方紅魔郷.exe', '*.exe')))


class CustomWindow(Window):
    def __init__(self, backend, width, height, fps_limit, frameskip):
        super().__init__(backend, width=width, height=height, fps_limit=fps_limit, frameskip=frameskip)
        self.keystate = 0

    def set_keystate(self, keystate):
        self.keystate = keystate

    def get_keystate(self):
        return self.keystate



class TouhouGym(gymnasium.Env):

    def __init__(
            self,
            game_path='./res/game/',
    ):
        self.gl_options = {
            'flavor': 'compatibility',
            'version': 2.1,
            'double-buffer': None,
            'frontend': 'glfw',
            'backend': 'opengl'
        }

        self.resource_path = abspath(game_path)
        self.fb_downscale_factor = 8
        self.channels = 1
        self.fps_limit = 60
        self.fb_greyscale = True
        self.input_shape = (y // self.fb_downscale_factor, x // self.fb_downscale_factor, self.channels)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.input_shape,
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(actions) - 1)
        self.rewards = 0

        self.characters = [0]
        self.continues = 0
        self.stage_num = random.randint(1, 6)
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

        self.window = CustomWindow(backend, Interface.width, Interface.height, fps_limit=self.fps_limit, frameskip=0)
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
        self.stage_num = random.randint(1, 6)
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
        self.game.players[0].lives = self.starting_lives
        self.rewards = 0

    def _get_obs(self):
        framebuffer = self.renderer.get_framebuffer(Interface.width, Interface.height, greyscale=self.fb_greyscale)
        img = np.frombuffer(framebuffer, dtype=np.uint8).reshape((Interface.height, Interface.width, self.channels))
        cropped = np.ascontiguousarray(img[offset_y:offset_y+y, offset_x:offset_x+x])
        scaled = np.flipud(tinyscaler.scale(cropped,
                                  (x // self.fb_downscale_factor, y // self.fb_downscale_factor)))
        return scaled

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
        except NextStage:
            terminated = True
        except GameOver:
            terminated = True

        is_dead = self.starting_lives > self.game.players[0].lives
        self.game.players[0].lives = 0
        reward = -1 if is_dead else self.game.players[0].rewards - self.rewards

        self.rewards = self.game.players[0].rewards

        observation = self._get_obs()

        return observation, reward, terminated, False, {}

    def close(self):
        self.window.set_runner(None)
        gc.collect()
