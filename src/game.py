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

x = 384
y = 448
current_score = 0
rewards = 0

def bullet_intersects_hitbox(player_x, player_y, hitbox, bullet_x, bullet_y, dx, dy, max_distance=50):
    x1, x2 = player_x - hitbox, player_x + hitbox
    y1, y2 = player_y - hitbox, player_y + hitbox

    epsilon = 1e-8
    dx = dx if dx != 0 else epsilon
    dy = dy if dy != 0 else epsilon

    tx1 = (x1 - bullet_x) / dx
    tx2 = (x2 - bullet_x) / dx
    ty1 = (y1 - bullet_y) / dy
    ty2 = (y2 - bullet_y) / dy

    # Ensure correct min/max intervals
    tmin_x, tmax_x = min(tx1, tx2), max(tx1, tx2)
    tmin_y, tmax_y = min(ty1, ty2), max(ty1, ty2)

    # Find global entry and exit points
    t_entry = max(tmin_x, tmin_y)
    t_exit = min(tmax_x, tmax_y)

    if t_exit < 0:
        return False, -1
    if t_entry > t_exit:
        return False, -1
    if t_entry > max_distance:
        return False, -1

    return True, np.sqrt((bullet_x - player_x) ** 2 + (bullet_y - player_y) ** 2)

def item_intersects_hitbox(player_x, player_y, hitbox, item_x, item_y, max_distance=590):
    """
    Check if a falling item intersects the player's hitbox within a vertical range.

    Parameters:
    - player_x, player_y: center of player's hitbox
    - hitbox: half-width/height of square hitbox (i.e., radius)
    - item_x, item_y: current item position
    - max_distance: max vertical distance to consider (optional)

    Returns:
    - (bool, float): (True if intersects, distance to item)
    """

    # Hitbox bounds
    x1, x2 = player_x - hitbox, player_x + hitbox

    # Check if item's x is within the player's horizontal hitbox
    if not (x1 <= item_x <= x2) or (player_y < item_y):
        return False, -1

    distance = np.sqrt((item_x - player_x) ** 2 + (item_y - player_y) ** 2)
    if distance > max_distance:
        return False, -1

    return True, distance

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
    window = Window(backend=backend, width=Interface.width, height=Interface.height, disable_render=True, fps_limit=60, frameskip=-1)
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
    game.players[0].lives = 500

    runner.load_game(game, game.background, game.std.bgms, None, None)


    def run_frame():
        while window.run_frame():
            print(game.lasers)
            # for bullet in game.bullets:
            #     intersect, distance = bullet_intersects_hitbox(game.players[0].x, game.players[0].y, game.players[0].sht.hitbox, bullet.x, bullet.y, bullet.dx, bullet.dy)
            #     if intersect:
            #         neg_reward = max(0.01, 1.0 - (distance / 50))
            #         print(neg_reward)
                    #reward -= neg_reward
            # for item in game.items:
            #    intersect, distance = item_intersects_hitbox(game.players[0].x, game.players[0].y, game.players[0].sht.item_hitbox, item.x, item.y)
            #    if intersect:
            #        print()
            #bullet_coords = np.array([(b.x, b.y, b.dx, b.dy, b.speed / 1000, normalize_radians(b.angle)) for b in game.bullets]) if game.bullets else np.empty((0, 6))
            #enemy_coords = np.array([(enm.x, enm.y, enm.angle, enm.speed / 1000, enm.rotation_speed / 1000, enm.acceleration / 1000) for enm in game.enemies]) if game.enemies else np.empty((0, 6))
            #item_coords = np.array([(i.x, i.y, -1, -1, -1, -1) for i in game.items]) if game.items else np.empty((0, 6))
            # target_x = 192 / x
            # target_y = 384 / y
            # target_pos = np.array([target_x, target_y])
            # player_pos = np.array([game.players[0].x / x, game.players[0].y / y])
            # distance_to_target = np.linalg.norm(player_pos - target_pos)
            # proximity_reward = -0.1 * distance_to_target  # Tune this scale
            #
            # is_dead = 0 > game.players[0].lives
            # reward = game.players[0].score - current_score
            # current_score = game.players[0].score
            # reward /= 100
            # if is_dead:
            #     reward -= 10
            # else:
            #     reward += 0.001  # reward for living
            #
            #
            # reward += proximity_reward
            # total_rewards += reward
            # print(f"{reward}, {total_rewards}")
            # global rewards
            # print(game.players[0].lives)
            # is_dead = 0 > game.players[0].lives
            # game.players[0].lives = 0
            # # reward for keeping towards (192, 384)
            # target_x = 192 / x
            # target_y = 384 / y
            # target_pos = np.array([target_x, target_y])
            # player_pos = np.array([game.players[0].x / x, game.players[0].y / y])
            # distance_to_target = np.linalg.norm(player_pos - target_pos)
            # proximity_reward = -0.1 * distance_to_target  # Tune this scale
            #
            # reward = -1 if is_dead else (game.players[0].rewards - rewards) + proximity_reward
            # rewards = game.players[0].rewards
            #print(reward)
            # player = game.players[0]
            # px, py = player.x, player.y
            # phalf_size = player.sht.hitbox
            # for bullet in game.bullets:
            #     intersect, distance = bullet_intersects_hitbox(px, py, phalf_size, bullet.x, bullet.y, bullet.dx, bullet.dy)
            #     if intersect:
            #         print(max(0.1, 1.0 - (distance / 50)))
            pass

    while True:
        try:
            run_frame()
            break
        except NextStage:
            stage_num += 1
            print("NEXT STAGE")
        except GameOver:
            print("GAME OVER")
            break
    window.set_runner(None)


if __name__ == '__main__':
    start()
