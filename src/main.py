import sys

from os.path import pathsep, abspath

from pytouhou.game import NextStage, GameOver
from pytouhou.ui.gamerunner import GameRunner
from pytouhou.utils.random import Random

from pytouhou.games.eosd.game import Game, Common
from pytouhou.games.eosd.interface import Interface
from pytouhou.resource.loader import Loader
from pytouhou.ui.opengl import backend
from pytouhou.ui.window import Window

import logging

import numpy as np

gl_options = {
    'flavor': 'compatibility',
    'version': 2.1,
    'double-buffer': None,
    'frontend': 'glfw',
    'backend': 'opengl'
}

game_data_locations = (pathsep.join(('CM.DAT', 'th06*_CM.DAT', '*CM.DAT', '*cm.dat')),
                       pathsep.join(('ST.DAT', 'th6*ST.DAT', '*ST.DAT', '*st.dat')),
                       pathsep.join(('IN.DAT', 'th6*IN.DAT', '*IN.DAT', '*in.dat')),
                       pathsep.join(('MD.DAT', 'th6*MD.DAT', '*MD.DAT', '*md.dat')),
                       pathsep.join(('102h.exe', '102*.exe', '東方紅魔郷.exe', '*.exe')))
resource_path = abspath('./res/game/')


# class Main:
#     def __init__(self):
#         self.characters = [np.random.randint(0, 3)]
#         self.continues = 0
#         self.stage_num = np.random.randint(1, 7)
#         self.rank = 3
#         self.difficulty = 16
#
#         self.resource_loader = None
#         self.game = None
#         self.prng = None
#         self.runner = None
#         self.interface = None
#         self.common = None
#         self.renderer = None
#         self.window = None
#
#         self.start()
#         self.reset()
#         self.run()
#
#     def start(self):
#         self.resource_loader = Loader(resource_path)
#         try:
#             self.resource_loader.scan_archives(game_data_locations)
#             backend.init(gl_options)
#         except IOError:
#             show_simple_message_box(u'Some data files were not found, did you forget the -p option?')
#             sys.exit(1)
#         except AssertionError as e:
#             logging.error(f'Backend failed to initialize: {e}')
#             sys.exit(1)
#
#         self.window = Window(backend, Interface.width, Interface.height, fps_limit=120, frameskip=0)
#         self.renderer = backend.GameRenderer(self.resource_loader, self.window)
#         self.common = Common(self.resource_loader, self.characters, self.continues)
#         self.interface = Interface(self.resource_loader, self.common.players[0])
#         self.common.interface = self.interface
#         self.runner = GameRunner(self.window, self.renderer, self.common, self.resource_loader)
#         self.window.set_runner(self.runner)
#
#     def reset(self):
#         if self.game is not None:
#             self.game.cleanup()
#
#         self.characters = [np.random.randint(0, 3)]
#         self.continues = 0
#         self.stage_num = np.random.randint(1, 7)
#         self.rank = 3
#         self.difficulty = 16
#
#         self.common = Common(self.resource_loader, self.characters, self.continues)
#         self.interface = Interface(self.resource_loader, self.common.players[0])
#         self.common.interface = self.interface
#
#         self.prng = Random()
#         self.game = Game(
#             resource_loader=self.resource_loader,
#             stage=self.stage_num,
#             rank=self.rank,
#             difficulty=self.difficulty,
#             common=self.common,
#             prng=self.prng
#         )
#
#         self.runner.load_game(self.game, self.game.background, self.game.std.bgms, None, None)
#         self.game.players[0].lives = 0
#
#     def step(self):
#         self.window.run_frame()
#
#     def run(self):
#         while True:
#             try:
#                 self.step()
#             except NextStage:
#                 self.stage_num += 1
#             except GameOver:
#                 self.reset()
#
#
# if __name__ == '__main__':
#     main = Main()

def start():
    characters = [0]
    continues = 0
    stage_num = 1
    rank = 3
    difficulty = 16


    resource_loader = Loader(resource_path)

    try:
        resource_loader.scan_archives(game_data_locations)
    except IOError:
        print(u'Some data files were not found, did you forget the -p option?')
        sys.exit(1)

    try:
        backend.init(gl_options)
    except AssertionError as e:
        logging.error(f'Backend failed to initialize: {e}')
        sys.exit(1)

    GameRenderer = backend.GameRenderer
    window = Window(backend, Interface.width, Interface.height, fps_limit=60, frameskip=0)
    common = Common(resource_loader, characters, continues)
    interface = Interface(resource_loader, common.players[0])
    common.interface = interface
    renderer = GameRenderer(resource_loader, window)
    runner = GameRunner(window, renderer, common, resource_loader)
    window.set_runner(runner)
    prng = Random()
    game = Game(
        resource_loader=resource_loader,
        stage=stage_num,
        rank=rank,
        difficulty=difficulty,
        common=common,
        prng=prng
    )
    game.players[0].lives = 0

    runner.load_game(game, game.background, game.std.bgms, None, None)

    def run_frame():
        while window.run_frame():
            pass

    while True:
        try:
            run_frame()
            break
        except NextStage:
            stage_num += 1
        except GameOver:
            break
    window.set_runner(None)


if __name__ == '__main__':
    start()
