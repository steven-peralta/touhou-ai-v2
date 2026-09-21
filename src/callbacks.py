import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

EPISODE_METRICS = ('hits', 'score', 'cleared', 'frames')


class EpisodeMetricsCallback(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_end(self):
        episodes = [ep for ep in self.model.ep_info_buffer if all(key in ep for key in EPISODE_METRICS)]
        if not episodes:
            return
        for key in EPISODE_METRICS:
            self.logger.record(f'rollout/ep_{key}_mean', float(np.mean([ep[key] for ep in episodes])))


class MetricsEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_metrics = {key: [] for key in EPISODE_METRICS}

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if not locals_['done']:
            return
        info = locals_['info']
        for key in EPISODE_METRICS:
            if key in info:
                self._episode_metrics[key].append(info[key])
                self.logger.record(f'eval/mean_ep_{key}', float(np.mean(self._episode_metrics[key])))

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._episode_metrics = {key: [] for key in EPISODE_METRICS}
        return super()._on_step()
