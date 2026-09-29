"""Per-term reward decomposition of a trained policy, per stage.

Runs the loaded policy stochastically (as during training) for a number of episodes on each
stage and reports the mean per-episode sum of every reward term, alongside hits, score, frames
and item pickups. Also summarises how much score reward the policy collects in bursts and around
hits, which is what decides whether "take a hit to grab an item shower" pays off.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor

from callbacks import EPISODE_METRICS
from touhou_gym import TouhouGym, REWARD_TERMS

FRAMES_PER_SECOND = 60


def _percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else 0.0


class EpisodeCollector:
    """Collects per-episode metrics and per-step score-term traces from evaluate_policy's callback."""

    def __init__(self, n_envs, window_steps):
        self.window_steps = max(window_steps, 1)
        self.episodes = []
        self._score_trace = [[] for _ in range(n_envs)]
        self._hit_steps = [[] for _ in range(n_envs)]
        self._last_hits = [0] * n_envs

    def __call__(self, locals_, globals_):
        i = locals_['i']
        info = locals_['info']
        step_reward = info.get('step_reward')
        if step_reward is not None:
            self._score_trace[i].append(float(step_reward['score']))
        hits = info.get('hits', 0)
        if hits > self._last_hits[i]:
            self._hit_steps[i].append(len(self._score_trace[i]) - 1)
        self._last_hits[i] = hits
        if not locals_['done']:
            return
        self.episodes.append(self._finish(i, info))
        self._score_trace[i] = []
        self._hit_steps[i] = []
        self._last_hits[i] = 0

    def _finish(self, i, info):
        trace = np.array(self._score_trace[i], dtype=np.float64)
        episode = {key: float(info.get(key, 0)) for key in EPISODE_METRICS}
        episode['total'] = float(sum(info.get(f'reward_{term}', 0.0) for term in REWARD_TERMS))
        window = self.window_steps
        if len(trace) >= window:
            sums = np.convolve(trace, np.ones(window), mode='valid')
        else:
            sums = np.array([trace.sum()]) if len(trace) else np.array([0.0])
        episode['score_burst_1s_max'] = float(sums.max())
        episode['score_step_p99'] = _percentile(trace, 99)
        episode['score_step_max'] = float(trace.max()) if len(trace) else 0.0
        near_hit = []
        for step in self._hit_steps[i]:
            near_hit.append(float(trace[max(0, step - window):step + window + 1].sum()))
        episode['score_near_hit'] = float(np.mean(near_hit)) if near_hit else 0.0
        episode['score_near_hit_max'] = float(np.max(near_hit)) if near_hit else 0.0
        return episode


# Columns of the per-stage summary, in print order.
REPORT_COLUMNS = (
    ['episodes', 'frames', 'hits', 'items', 'score']
    + [f'reward_{term}' for term in REWARD_TERMS]
    + ['total', 'score_burst_1s_max', 'score_near_hit', 'score_near_hit_max', 'score_step_p99', 'score_step_max']
)


def summarize(episodes):
    summary = {'episodes': len(episodes)}
    for column in REPORT_COLUMNS:
        if column == 'episodes':
            continue
        summary[column] = float(np.mean([ep[column] for ep in episodes])) if episodes else 0.0
    return summary


def format_table(rows):
    """rows: list of dicts with a 'stage' key plus REPORT_COLUMNS."""
    columns = ['stage'] + list(REPORT_COLUMNS)
    widths = {col: max(len(col), 10) for col in columns}

    def fmt(col, value):
        if col in ('stage', 'episodes'):
            return str(int(value))
        if col in ('frames', 'score', 'items', 'hits'):
            return f'{value:,.1f}'
        return f'{value:+.2f}'

    lines = ['  '.join(col.rjust(widths[col]) for col in columns)]
    for row in rows:
        lines.append('  '.join(fmt(col, row[col]).rjust(widths[col]) for col in columns))
    return '\n'.join(lines)


def make_env_fn(stage, mortal, game_res_path, action_repeat, hit_penalty, score_reward_scale, score_reward_cap):
    def make():
        return TouhouGym(
            disable_render=True,
            stage_num=stage,
            fps_limit=-1,
            unlock_fps=True,
            game_path=game_res_path,
            action_repeat=action_repeat,
            mortal=mortal,
            hit_penalty=hit_penalty,
            score_reward_scale=score_reward_scale,
            score_reward_cap=score_reward_cap,
        )
    return make


def run_stage(model, stage, n_episodes, n_envs, frame_stack_size, action_repeat, mortal, game_res_path,
              hit_penalty, score_reward_scale, score_reward_cap, env_fn_factory=make_env_fn, vec_env_cls=SubprocVecEnv):
    n_envs = max(1, min(n_envs, n_episodes))
    env_fn = env_fn_factory(stage, mortal, game_res_path, action_repeat, hit_penalty, score_reward_scale, score_reward_cap)
    kwargs = {'start_method': 'spawn'} if vec_env_cls is SubprocVecEnv else {}
    env = vec_env_cls([env_fn for _ in range(n_envs)], **kwargs)
    env = VecFrameStack(env, n_stack=frame_stack_size)
    env = VecMonitor(env, info_keywords=EPISODE_METRICS)
    collector = EpisodeCollector(n_envs, window_steps=FRAMES_PER_SECOND // max(action_repeat, 1))
    try:
        evaluate_policy(model, env, n_eval_episodes=n_episodes, deterministic=False, callback=collector)
    finally:
        env.close()
    return collector.episodes


def write_report(output_dir, report):
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    base = os.path.join(reports_dir, f"reward-report-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    with open(base + '.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open(base + '.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['stage'] + list(REPORT_COLUMNS))
        writer.writeheader()
        for row in report['stages']:
            writer.writerow({col: row[col] for col in ['stage'] + list(REPORT_COLUMNS)})
    return base + '.json', base + '.csv'


def reward_report(
        load_from_checkpoint,
        stages,
        n_episodes,
        n_envs,
        output_dir,
        frame_stack_size=1,
        action_repeat=2,
        game_res_path='./res/game/',
        device='cuda',
        mortal=False,
        hit_penalty=2.0,
        score_reward_scale=0.1,
        score_reward_cap=None,
        model=None,
        env_fn_factory=make_env_fn,
        vec_env_cls=SubprocVecEnv,
):
    if model is None:
        model = RecurrentPPO.load(load_from_checkpoint, device=device)

    config = dict(
        checkpoint=load_from_checkpoint,
        stages=list(stages),
        episodes_per_stage=n_episodes,
        mortal=mortal,
        action_repeat=action_repeat,
        frame_stack=frame_stack_size,
        hit_penalty=hit_penalty,
        score_reward_scale=score_reward_scale,
        score_reward_cap=score_reward_cap,
        deterministic=False,
    )
    print(f"Reward report for {load_from_checkpoint}: {n_episodes} stochastic episodes per stage on stages {list(stages)}, "
          f"{'mortal' if mortal else 'invincible'} mode, hit_penalty={hit_penalty}, score_reward_scale={score_reward_scale}, "
          f"score_reward_cap={score_reward_cap}")

    rows, per_episode = [], {}
    started = time.time()
    for stage in stages:
        stage_started = time.time()
        episodes = run_stage(model, stage, n_episodes, n_envs, frame_stack_size, action_repeat, mortal, game_res_path,
                             hit_penalty, score_reward_scale, score_reward_cap, env_fn_factory, vec_env_cls)
        row = {'stage': stage, **summarize(episodes)}
        rows.append(row)
        per_episode[str(stage)] = episodes
        print(f"stage {stage}: {len(episodes)} episodes in {time.time() - stage_started:.0f}s")
        sys.stdout.flush()

    report = {
        'config': config,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'seconds': time.time() - started,
        'stages': rows,
        'episodes': per_episode,
    }
    json_path, csv_path = write_report(output_dir, report)

    print()
    print("Mean per-episode sums (reward_* columns are the reward terms; total is their sum, i.e. the episode return).")
    print("score_burst_1s_max: largest score reward collected in any 1 s window; score_near_hit: score reward within ±1 s of a hit.")
    print(format_table(rows))
    print(f"\nWrote {json_path} and {csv_path} in {report['seconds']:.0f}s")
    return report
