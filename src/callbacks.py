import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from touhou_gym import REWARD_TERMS

# Per-episode values copied from the env info dict at episode end and logged as
# rollout/ep_<key>_mean and eval/mean_ep_<key>. The reward_<term> keys are the
# per-episode sums of each reward term, so their sum matches ep_rew_mean.
EPISODE_METRICS = ('hits', 'score', 'cleared', 'frames', 'items') + tuple(f'reward_{term}' for term in REWARD_TERMS)


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, stream=None):
        super().__init__()
        self.stream = stream

    def _on_step(self):
        if self.stream is not None and self.n_calls % 20 == 0:
            self.stream.set_progress(self.model.num_timesteps, self._steps_to_eval())
        return True

    def _on_rollout_start(self):
        if self.stream is not None:
            self.stream.set_phase('collecting rollout')

    def _on_rollout_end(self):
        episodes = [ep for ep in self.model.ep_info_buffer if all(key in ep for key in EPISODE_METRICS)]
        if episodes:
            for key in EPISODE_METRICS:
                self.logger.record(f'rollout/ep_{key}_mean', float(np.mean([ep[key] for ep in episodes])))
        if self.stream is not None:
            self.stream.update_metrics({'time/total_timesteps': self.model.num_timesteps})
            self.stream.set_phase('policy update')

    def _on_training_start(self):
        if self.stream is None:
            return
        logger = self.logger
        original_record = logger.record
        stream = self.stream

        def record(key, value, exclude=None):
            original_record(key, value, exclude)
            if isinstance(value, (int, float, np.floating, np.integer)):
                stream.update_metrics({key: float(value)})

        logger.record = record

        buffer = self.model.rollout_buffer
        original_get = buffer.get
        n_epochs = self.model.n_epochs
        epoch = [0]

        def get(batch_size=None):
            epoch[0] = epoch[0] % n_epochs + 1
            stream.set_phase(f'policy update, epoch {epoch[0]}/{n_epochs}')
            return original_get(batch_size)

        buffer.get = get

    def _steps_to_eval(self):
        eval_callback = getattr(self, 'eval_callback', None)
        if eval_callback is None or eval_callback.eval_freq <= 0:
            return 0
        if self.stream is not None:
            self.stream.update_metrics({'_eval_span': eval_callback.eval_freq * self.training_env.num_envs})
        calls_left = eval_callback.eval_freq - (eval_callback.n_calls % eval_callback.eval_freq)
        return calls_left * self.training_env.num_envs


class MetricsEvalCallback(EvalCallback):
    def __init__(self, *args, stream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_metrics = {key: [] for key in EPISODE_METRICS}
        self.stream = stream
        self._eval_episode = 0

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if self.stream is not None and locals_['i'] == 0:
            if not locals_['done']:
                frame = self.eval_env.env_method('render', indices=0)[0]
                if frame is not None:
                    self.stream.push_frame(np.ascontiguousarray(frame[..., :3]))
            else:
                self._eval_episode += 1
                self.stream.set_status(f'eval stage {self._eval_stage()}: episode {self._eval_episode}/{self.n_eval_episodes} done, {locals_["info"].get("hits", "?")} hits')
        if not locals_['done']:
            return
        info = locals_['info']
        for key in EPISODE_METRICS:
            if key in info:
                self._episode_metrics[key].append(info[key])
                self.logger.record(f'eval/mean_ep_{key}', float(np.mean(self._episode_metrics[key])))

    def _eval_stage(self):
        try:
            return self.eval_env.get_attr('stage_num', indices=0)[0]
        except Exception:
            return '?'

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._episode_metrics = {key: [] for key in EPISODE_METRICS}
            self._eval_episode = 0
            if self.stream is not None:
                self.stream.set_phase(f'evaluating on stage {self._eval_stage()}')
                self.stream.set_status(f'eval at {self.model.num_timesteps:,} steps')
        continue_training = super()._on_step()
        if self.stream is not None and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.stream.record_eval({k: v for k, v in self.stream.metrics.items() if k.startswith('eval/')})
        return continue_training
