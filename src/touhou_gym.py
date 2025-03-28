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
#FOCUS = 4

actions = [
    SHOOT,
    UP,
    UP | LEFT,
    UP | LEFT | SHOOT,
    #UP | LEFT | SHOOT | FOCUS,
    UP | RIGHT,
    UP | RIGHT | SHOOT,
    #UP | RIGHT | SHOOT | FOCUS,
    DOWN,
    DOWN | LEFT,
    DOWN | LEFT | SHOOT,
    #DOWN | LEFT | SHOOT | FOCUS,
    DOWN | RIGHT,
    DOWN | RIGHT | SHOOT,
    #DOWN | RIGHT | SHOOT | FOCUS,
    LEFT,
    LEFT | SHOOT,
    #LEFT | SHOOT | FOCUS,
    RIGHT,
    RIGHT | SHOOT,
    #RIGHT | SHOOT | FOCUS
]

game_data_locations = (pathsep.join(('CM.DAT', 'th06*_CM.DAT', '*CM.DAT', '*cm.dat')),
                       pathsep.join(('ST.DAT', 'th6*ST.DAT', '*ST.DAT', '*st.dat')),
                       pathsep.join(('IN.DAT', 'th6*IN.DAT', '*IN.DAT', '*in.dat')),
                       pathsep.join(('MD.DAT', 'th6*MD.DAT', '*MD.DAT', '*md.dat')),
                       pathsep.join(('102h.exe', '102*.exe', '東方紅魔郷.exe', '*.exe')))


def closest_point(points, point):
    if points.size == 0:
        return -1, -1, -1, -1, -1, -1, -1  # Default value if no objects exist

    px, py = point
    min_distance_sq = np.inf
    closest = None

    for p in points:
        n_x = p[0]
        n_y = p[1]
        dx = n_x - px
        dy = n_y - py
        distance_sq = np.sqrt(dx ** 2 + dy ** 2)
        if distance_sq < min_distance_sq:
            min_distance_sq = distance_sq
            closest = (n_x, n_y, distance_sq, p[2], p[3], p[4], p[5])

    # Normalize coordinates and distance
    if closest:
        return closest[0], closest[1], closest[2], closest[3], closest[4], closest[5], closest[6]  # Normalize

    return -1, -1, -1, -1, -1, -1, -1  # Fallback in case of error

def item_intersects_hitbox(player_x, player_y, hitbox, item_x, item_y, max_distance=448):
    x1, x2 = player_x - hitbox, player_x + hitbox

    # Check if item's x is within the player's horizontal hitbox
    if not (x1 <= item_x <= x2) or (player_y < item_y):
        return False, -1

    distance = np.sqrt((item_x - player_x) ** 2 + (item_y - player_y) ** 2)
    if distance > max_distance:
        return False, -1

    return True, distance


def bullet_intersects_hitbox(player_x, player_y, hitbox, bullet_x, bullet_y, dx, dy, max_distance=590):
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


class CustomWindow(Window):
    def __init__(self, backend, width, height, fps_limit, frameskip, unlock_fps):
        super().__init__(backend, width=width, height=height, fps_limit=fps_limit, frameskip=frameskip, no_delay=unlock_fps)
        self.keystate = 0

    def set_keystate(self, keystate):
        self.keystate = keystate

    def get_keystate(self):
        return self.keystate



class TouhouGym(gymnasium.Env):

    def __init__(
            self,
            game_path='./res/game/',
            image_scale=8,
            greyscale=True,
            stage_num=1,
            random_stage=False,
            fps_limit=-1,
            unlock_fps=True,
    ):
        self.gl_options = {
            'flavor': 'compatibility',
            'version': 2.1,
            'double-buffer': None,
            'frontend': 'glfw',
            'backend': 'opengl'
        }
        self.render_mode = 'rgb_array'
        self.resource_path = abspath(game_path)
        self.fb_downscale_factor = image_scale
        self.channels = 1 if greyscale else 3
        self.fps_limit = fps_limit
        self.unlock_fps = unlock_fps
        self.fb_greyscale = greyscale
        self.input_shape = (y // self.fb_downscale_factor, x // self.fb_downscale_factor, self.channels)
        self.observation_space = spaces.Dict({
            'features': spaces.Box(low=-np.inf, high=np.inf,shape=(19,)),
            'image': spaces.Box(
                low=0,
                high=255,
                shape=self.input_shape,
                dtype=np.uint8
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

    def _agent_eyes(self):
        framebuffer = self.renderer.get_framebuffer(Interface.width, Interface.height, greyscale=self.fb_greyscale)
        img = np.frombuffer(framebuffer, dtype=np.uint8).reshape((Interface.height, Interface.width, self.channels))
        cropped = np.ascontiguousarray(img[offset_y:offset_y + y, offset_x:offset_x + x])
        scaled = np.flipud(tinyscaler.scale(cropped, (x // self.fb_downscale_factor, y // self.fb_downscale_factor)))
        # Convert to image
        if self.fb_greyscale:
            pil_img = Image.fromarray(scaled.squeeze(), mode='L')
        else:
            pil_img = Image.fromarray(scaled, mode='RGB')

        # Save the image
        pil_img.save(f"/Users/steven/Development/touhou-ai-v2/image/agent_view_{self.game.frame}.png")

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

        self.window = CustomWindow(backend, Interface.width, Interface.height, fps_limit=self.fps_limit, frameskip=0, unlock_fps=self.unlock_fps)
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
        self.game.players[0].lives = self.starting_lives
        self.current_score = 0
        self.rewards = 0

    def _get_obs(self):
        framebuffer = self.renderer.get_framebuffer(Interface.width, Interface.height, greyscale=self.fb_greyscale)
        img = np.frombuffer(framebuffer, dtype=np.uint8).reshape((Interface.height, Interface.width, 4))
        cropped = np.ascontiguousarray(img[offset_y:offset_y + y, offset_x:offset_x + x])
        scaled = np.flipud(tinyscaler.scale(cropped, (x // self.fb_downscale_factor, y // self.fb_downscale_factor), mode='bilinear'))
        scaled = scaled[:, :, :3]

        # Convert entities to NumPy arrays
        bullet_coords = np.array([(b.x, b.y, b.dx, b.dy, b.speed, b.angle) for b in
                                  self.game.bullets]) if self.game.bullets else np.empty((0, 6))
        enemy_coords = np.array(
            [(enm.x, enm.y, enm.angle, enm.speed, enm.rotation_speed, enm.acceleration) for enm in
             self.game.enemies]) if self.game.enemies else np.empty((0, 6))
        item_coords = np.array([(i.x, i.y, -1, -1, -1, -1) for i in self.game.items]) if self.game.items else np.empty((0, 6))
        player_coords = np.array([self.game.players[0].x, self.game.players[0].y])

        # Compute closest entities
        closest_bullet = closest_point(bullet_coords, player_coords)
        closest_enemy = closest_point(enemy_coords, player_coords)
        closest_item = closest_point(item_coords, player_coords)[:2]

        is_dead = 1 if self.starting_lives > self.game.players[0].lives else 0

        # Stack all features into a single array (fixed size)
        features = np.hstack([player_coords, closest_bullet, closest_enemy, closest_item, is_dead])
        return {'image': scaled, 'features': features}

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
        observation = self._get_obs()
        is_dead = self.starting_lives > self.game.players[0].lives
        self.game.players[0].lives = 0

        # reward for keeping towards (192, 384)
        target_x = 192
        target_y = 384
        target_pos = np.array([target_x, target_y])
        player_pos = np.array([self.game.players[0].x, self.game.players[0].y])
        distance_to_target = np.linalg.norm(player_pos - target_pos)
        proximity_reward = -0.01 * distance_to_target  # Tune this scale

        reward = self.game.players[0].rewards - self.rewards
        self.rewards = self.game.players[0].rewards

        reward += proximity_reward

        if is_dead:
            reward -= 10
            terminated = True
            return observation, reward, terminated, False, {}
        else:
            reward += 0.01 # reward for staying alive

            # subtract for being in vector of bullet max at 50 pixels
            for bullet in self.game.bullets:
                intersect, distance = bullet_intersects_hitbox(self.game.players[0].x, self.game.players[0].y, self.game.players[0].sht.hitbox, bullet.x, bullet.y, bullet.dx, bullet.dy)
                if intersect:
                    reward -= max(0.01, 1.0 - (distance / 50)) * 1.25

            # add for being in the x coordinate of an item
            for item in self.game.items:
                intersect, distance = item_intersects_hitbox(self.game.players[0].x, self.game.players[0].y, self.game.players[0].sht.item_hitbox, item.x, item.y)
                if intersect:
                    reward += max(0.01, 1.0 - (distance / 590))

            return observation, reward, terminated, False, {}



    def close(self):
        self.window.set_runner(None)
        gc.collect()
