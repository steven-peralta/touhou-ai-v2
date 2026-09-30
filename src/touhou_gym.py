import math
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

from gym_utils import entity_array, closest_entity, bullet_intersects_hitbox, GAME_WIDTH, GAME_HEIGHT

UP = 16
DOWN = 32
LEFT = 64
RIGHT = 128
SHOOT = 1
BOMB = 2
FOCUS = 4

DIRECTIONS = [0, UP, DOWN, LEFT, RIGHT, UP | LEFT, UP | RIGHT, DOWN | LEFT, DOWN | RIGHT]

MAX_BULLETS = 640
MAX_LASERS = 128
MAX_ENEMIES = 64
MAX_ITEMS = 20

PLAYER_FEATURES = 5   # hitbox extents (4) + remaining lives fraction
BULLET_FEATURES = 10  # present, rel xy, near xy, velocity, hitbox, launching flag
LASER_FEATURES = 11   # present, rel xy, near xy, direction, half width, started, starting, start progress
ENEMY_FEATURES = 5    # present, rel xy, hitbox half sizes
ITEM_FEATURES = 4     # present, rel xy, type

VELOCITY_SCALE = 8.0
NEAR_SCALE = 48.0
HITBOX_SCALE = 32.0

SCORE_REWARD_SCALE = 0.1
HIT_PENALTY = 2.0          # per hit that leaves lives to spare (invincible envs: every hit)
FATAL_HIT_PENALTY = 5.0    # for the hit that ends the episode in envs with finite lives
INVINCIBLE_LIVES = 9999
DANGER_PENALTY_SCALE = 0.1
DANGER_PENALTY_CAP = 5.0
STILL_PENALTY = 0.05
STILL_FRAMES = 60
EDGE_PENALTY = 0.02
EDGE_MARGIN = 32.0

# Reward terms, each summed per episode and reported in the step info dict as reward_<term>.
REWARD_TERMS = ('score', 'hits', 'danger', 'still', 'edge')

BULLET_LAUNCHING = 0
BULLET_CANCELLED = 2
LASER_STARTING = 0
LASER_STARTED = 1

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

    def get_human_keystate(self):
        return super().get_keystate()



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
            lives=0,
            action_repeat=2,
            hit_penalty=HIT_PENALTY,
            fatal_hit_penalty=FATAL_HIT_PENALTY,
            score_reward_scale=SCORE_REWARD_SCALE,
            score_reward_cap=None,
            danger_weighted_score=False,
    ):
        """
        lives: hits the episode survives before it ends. 0 = invincible (the episode never ends on a
        hit), 1 = mortal (the first hit ends it), N = the N-th hit ends it. Each hit that leaves
        lives to spare costs hit_penalty; the hit that ends the episode costs fatal_hit_penalty.
        """
        self.gl_options = {
            'flavor': 'compatibility',
            'version': 2.1,
            'double-buffer': None,
            'frontend': 'glfw',
            'backend': 'opengl'
        }
        self.disable_render = disable_render
        self.lives = int(lives)
        self.render_mode = 'rgb_array'
        self.resource_path = abspath(game_path)
        self.fps_limit = fps_limit
        self.unlock_fps = unlock_fps

        self.action_repeat = action_repeat
        self.hit_penalty = hit_penalty
        self.fatal_hit_penalty = fatal_hit_penalty
        self.score_reward_scale = score_reward_scale
        self.score_reward_cap = score_reward_cap
        self.danger_weighted_score = danger_weighted_score
        self.expected_lives = self._initial_lives()

        def box(*shape):
            return spaces.Box(low=-1, high=1, shape=shape, dtype=np.float32)

        self.observation_space = spaces.Dict({
            'game_player': box(PLAYER_FEATURES),
            'game_stage': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'game_boss': box(3),
            'game_closest_bullet': box(BULLET_FEATURES),
            'game_closest_item': box(ITEM_FEATURES),
            'game_closest_enemy': box(ENEMY_FEATURES),
            'pife_game_bullets': box(MAX_BULLETS, BULLET_FEATURES),
            'pife_game_lasers': box(MAX_LASERS, LASER_FEATURES),
            'pife_game_enemies': box(MAX_ENEMIES, ENEMY_FEATURES),
            'pife_game_items': box(MAX_ITEMS, ITEM_FEATURES),
        })

        # [direction(9: neutral/up/down/left/right/diagonals), shoot(2), focus(2)]
        self.action_space = spaces.MultiDiscrete([len(DIRECTIONS), 2, 2])
        self.current_score = 0
        self.last_graze = 0
        self.still_frames = 0
        self.episode_hits = 0
        self.episode_frames = 0
        self.episode_items = 0
        self.last_items = 0
        self.episode_reward = self._zero_terms()
        self.step_reward = self._zero_terms()
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
        self.quit_requested = False

        self._start()

    def render(self):
        if self.disable_render:
            return None
        framebuffer = self.renderer.get_framebuffer(Interface.width, Interface.height, greyscale=False)
        img = np.frombuffer(framebuffer, dtype=np.uint8).reshape((Interface.height, Interface.width, 4))
        if self.renderer.get_framebuffer_top_down():
            return img
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
            self.renderer.set_full_redraw(True)
            self.runner = GameRunner(self.window, self.renderer, self.common, self.resource_loader)
            self.window.set_runner(self.runner)
            self.runner.load_game(self.game, self.game.background, self.game.std.bgms, None, None)

        # The engine ends the game when lives drops below 0, so a budget of N hits is N - 1 engine lives.
        self.expected_lives = self._initial_lives()
        self.game.players[0].lives = self.expected_lives

        # Fast-forward past the empty pre-spawn phase
        while len(self.game.enemies) == 0 and len(self.game.bullets) == 0:
            self.game.run_iter([0])

        self.current_score = self.game.players[0].score
        self.last_graze = self.game.players[0].graze
        self.still_frames = 0
        self.episode_hits = 0
        self.episode_frames = 0
        self.episode_items = 0
        self.last_items = self.game.players[0].rewards
        self.episode_reward = self._zero_terms()
        self.step_reward = self._zero_terms()
        self.last_px = self.game.players[0].x
        self.last_py = self.game.players[0].y

    @staticmethod
    def _zero_terms():
        return {term: 0.0 for term in REWARD_TERMS}

    def _initial_lives(self):
        return INVINCIBLE_LIVES if self.lives <= 0 else self.lives - 1

    def _lives_fraction(self):
        """Remaining hits the episode survives, as a fraction of the budget (1.0 when invincible)."""
        if self.lives <= 0:
            return 1.0
        return max(self.expected_lives + 1, 0) / self.lives

    def _live_bullets(self):
        return [b for b in self.game.bullets if b.state != BULLET_CANCELLED][:MAX_BULLETS]

    def _get_obs(self):
        player = self.game.players[0]
        px, py = player.x, player.y
        h = player.sht.hitbox
        player_np = np.array(
            [(px + h) / GAME_WIDTH, (px - h) / GAME_WIDTH,
             (py + h) / GAME_HEIGHT, (py - h) / GAME_HEIGHT,
             self._lives_fraction()],
            dtype=np.float32)

        def rel(x, y):
            return (x - px) / GAME_WIDTH, (y - py) / GAME_HEIGHT

        def near(x, y):
            return (x - px) / NEAR_SCALE, (y - py) / NEAR_SCALE

        def bullet_row(b):
            hw, hh = b.get_hitbox()
            return (1.0, *rel(b.x, b.y), *near(b.x, b.y), b.dx / VELOCITY_SCALE, b.dy / VELOCITY_SCALE,
                    hw / HITBOX_SCALE, hh / HITBOX_SCALE, float(b.state == BULLET_LAUNCHING))

        def laser_row(laser):
            x0, y0, x1, y1, half_width, state = laser.get_segment()
            sx, sy = x1 - x0, y1 - y0
            length_sq = sx * sx + sy * sy
            t = 0.0 if length_sq == 0 else min(1.0, max(0.0, ((px - x0) * sx + (py - y0) * sy) / length_sq))
            nx, ny = x0 + t * sx, y0 + t * sy
            if state == LASER_STARTING and laser.start_duration > 0:
                start_progress = min(laser.frame / laser.start_duration, 1.0)
            else:
                start_progress = 1.0
            return (1.0, *rel(nx, ny), *near(nx, ny), math.cos(laser.angle), math.sin(laser.angle),
                    half_width / HITBOX_SCALE, float(state == LASER_STARTED), float(state == LASER_STARTING),
                    start_progress)

        bullets_np = entity_array(self._live_bullets(), MAX_BULLETS, BULLET_FEATURES, bullet_row)
        lasers_np = entity_array(self.game.lasers, MAX_LASERS, LASER_FEATURES, laser_row)
        def enemy_row(e):
            hw, hh = e.get_hitbox()
            return (1.0, *rel(e.x, e.y), hw / HITBOX_SCALE, hh / HITBOX_SCALE)

        enemies_np = entity_array(self.game.enemies, MAX_ENEMIES, ENEMY_FEATURES, enemy_row)
        items_np = entity_array(self.game.items, MAX_ITEMS, ITEM_FEATURES,
                                lambda i: (1.0, *rel(i.x, i.y), i._type / 6.0))

        boss = self.game.boss
        boss_np = np.array((1.0, *rel(boss.x, boss.y)) if boss else (0.0, 0.0, 0.0), dtype=np.float32)
        boss_np = np.clip(boss_np, -1, 1)

        stage_np = np.array([self.stage_num / 6.0], dtype=np.float32)

        return {
            'game_player': player_np,
            'game_stage': stage_np,
            'game_boss': boss_np,
            'game_closest_bullet': closest_entity(bullets_np),
            'game_closest_item': closest_entity(items_np),
            'game_closest_enemy': closest_entity(enemies_np),
            'pife_game_bullets': bullets_np,
            'pife_game_lasers': lasers_np,
            'pife_game_enemies': enemies_np,
            'pife_game_items': items_np
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._reset(seed)

        observation = self._get_obs()

        return observation, {}

    def _get_raw_bullet_array(self):
        bullets = self._live_bullets()
        bullet_array = np.full((MAX_BULLETS, 4), -1, dtype=np.float32)
        if bullets:
            bullet_array[:len(bullets)] = [(b.x, b.y, b.dx, b.dy) for b in bullets]
        return bullet_array

    def _script_finished(self):
        return all(runner.instruction_pointer >= len(runner._main) for runner in self.game.ecl_runners)

    def _resume_stalled_boss(self):
        boss = self.game.boss
        if boss is None or boss.damageable or boss.life != 1 or boss.timeout != -1:
            return
        if boss.boss_callback and self._script_finished():
            self.game.msg_wait = False
            boss.boss_callback.fire()

    def _stage_cleared(self):
        return self._script_finished() and not self.game.boss and not self.game.enemies

    def _run_frame(self, keystate):
        try:
            if self.disable_render:
                self.game.run_iter([keystate])
            else:
                self.window.set_keystate(keystate)
                if not self.window.run_frame():
                    self.quit_requested = True
        except NextStage:
            return True, True
        except GameOver:
            return True, False
        return False, False

    def _frame_reward(self):
        """Compute this frame's reward and return (reward, terminated).

        Each term is also accumulated into self.step_reward (reset every agent step) and
        self.episode_reward (reset every episode), keyed by REWARD_TERMS.
        """
        terminated = False
        player = self.game.players[0]
        terms = self._zero_terms()

        # Detect hit via lives dropping; extends and 1ups never add to the budget
        was_hit = player.lives < self.expected_lives
        fatal = was_hit and player.lives < 0
        if self.lives <= 0:
            player.lives = self.expected_lives
        else:
            self.expected_lives = player.lives = min(player.lives, self.expected_lives)

        # Normalized score delta as base reward, with graze contribution removed
        score_delta = player.score - self.current_score
        graze_delta = player.graze - self.last_graze
        self.current_score = player.score
        self.last_graze = player.graze

        # Gradient bullet danger
        valid_hits, dists = bullet_intersects_hitbox(player.x, player.y, player.sht.hitbox, self._get_raw_bullet_array())
        danger_score = 0.0
        if np.any(valid_hits):
            threatening_dists = dists[valid_hits]
            danger_score = float(np.sum(1.0 / (threatening_dists + 1.0)))
            terms['danger'] = -DANGER_PENALTY_SCALE * min(danger_score, DANGER_PENALTY_CAP)

        real_score_delta = score_delta - (graze_delta * 500)
        score_term = self.score_reward_scale * math.log1p(max(real_score_delta, 0) / 1000.0)
        if self.score_reward_cap is not None:
            score_term = min(score_term, self.score_reward_cap)
        if self.danger_weighted_score:
            # pay less for pickups made under fire
            score_term *= 1.0 - min(danger_score, 1.0)
        terms['score'] = score_term

        # Score-yielding item pickups (the engine counts them while the player is alive)
        self.episode_items += player.rewards - self.last_items
        self.last_items = player.rewards

        px, py = player.x, player.y

        # Hit handling: the hit that uses up the last life ends the episode
        if was_hit:
            self.episode_hits += 1
            if fatal:
                terms['hits'] = -self.fatal_hit_penalty
                terminated = True
            else:
                terms['hits'] = -self.hit_penalty

        # Stillness penalty: penalize staying in the same position for >60 frames
        moved = abs(px - self.last_px) > 1.0 or abs(py - self.last_py) > 1.0
        if moved:
            self.still_frames = 0
        else:
            self.still_frames += 1
        if self.still_frames > STILL_FRAMES:
            terms['still'] = -STILL_PENALTY

        # Edge penalty: penalize top, left, and right edges (bottom is okay)
        edge = 0.0
        if py < EDGE_MARGIN:  # too close to top
            edge -= EDGE_PENALTY
        if px < EDGE_MARGIN:  # too close to left
            edge -= EDGE_PENALTY
        if px > GAME_WIDTH - EDGE_MARGIN:  # too close to right
            edge -= EDGE_PENALTY
        terms['edge'] = edge

        self.last_px = px
        self.last_py = py

        reward = 0.0
        for term, value in terms.items():
            value = float(value)
            reward += value
            self.step_reward[term] += value
            self.episode_reward[term] += value

        return reward, terminated

    def step(self, action):
        direction, shoot, focus = action
        keystate = DIRECTIONS[direction]
        if shoot: keystate |= SHOOT
        if focus: keystate |= FOCUS

        reward = 0.0
        terminated = False
        cleared = False
        self.step_reward = self._zero_terms()

        for _ in range(self.action_repeat):
            try:
                terminated, cleared = self._run_frame(keystate)
            except AttributeError:
                # Engine can crash on boss_callback when boss is None during transitions
                terminated = True

            # Skip cutscenes/dialogue by fast-forwarding with shoot pressed
            while self.game.msg_wait and not terminated:
                try:
                    terminated, cleared = self._run_frame(SHOOT)
                except AttributeError:
                    # Engine can crash accessing boss_callback during dialogue transitions
                    break

            if not terminated:
                self._resume_stalled_boss()

            if not terminated and self._stage_cleared():
                terminated = True
                cleared = True

            self.episode_frames += 1
            frame_reward, died = self._frame_reward()
            reward += frame_reward
            terminated = terminated or died
            if terminated or self.quit_requested:
                break

        info = {
            'quit': self.quit_requested,
            'hits': self.episode_hits,
            'score': self.game.players[0].score,
            'cleared': int(cleared),
            'frames': self.episode_frames,
            'items': self.episode_items,
            'score_per_life': self.game.players[0].score / (self.episode_hits + 1),
            # episode tags for per-stage / per-env-type aggregation
            'stage': self.stage_num,
            'lives': self.lives,
            # this step's reward broken down by term
            'step_reward': dict(self.step_reward),
        }
        # per-episode sums of each reward term (sum over terms == episode return)
        for term in REWARD_TERMS:
            info[f'reward_{term}'] = self.episode_reward[term]

        return self._get_obs(), reward, terminated, False, info

    def human_action(self):
        keystate = self.window.get_human_keystate()
        direction = 0
        if bool(keystate & UP) != bool(keystate & DOWN):
            direction |= keystate & (UP | DOWN)
        if bool(keystate & LEFT) != bool(keystate & RIGHT):
            direction |= keystate & (LEFT | RIGHT)
        return np.array([DIRECTIONS.index(direction), int(bool(keystate & SHOOT)), int(bool(keystate & FOCUS))], dtype=np.int64)

    def close(self):
        if self.window is not None:
            self.window.set_runner(None)
        gc.collect()
