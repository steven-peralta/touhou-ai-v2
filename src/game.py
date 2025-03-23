import math
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
