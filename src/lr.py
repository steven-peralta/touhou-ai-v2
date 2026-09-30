"""Learning-rate schedules and the KL-adaptive learning-rate controller."""
from stable_baselines3.common.callbacks import BaseCallback


def linear_schedule(initial_value, min_value=0.0):
    def func(progress_remaining):
        return max(progress_remaining * initial_value, min_value)
    return func


def anchored_schedule(level, progress_at_start=1.0, decay='constant'):
    """LR schedule anchored to the progress at which training (re)started.

    SB3 evaluates schedules on progress_remaining = 1 - steps / total_steps, so a plain linear
    decay resumed at 60% progress would start at 0.4x the flag value. This one returns `level`
    at progress_at_start and, for decay='linear', decays to 0 at the end of total_steps;
    decay='constant' always returns `level`.

    It is a closure rather than a class instance on purpose: checkpoints pickle the model's
    schedule, and cloudpickle stores closures by value, so a checkpoint loads without this module.
    """
    if decay not in ('constant', 'linear'):
        raise ValueError(f"unknown lr schedule '{decay}'")
    level = float(level)
    p0 = max(float(progress_at_start), 1e-6)

    if decay == 'constant':
        def schedule(progress_remaining):
            return level
    else:
        def schedule(progress_remaining):
            return level * max(min(progress_remaining / p0, 1.0), 0.0)
    return schedule


def progress_at_start(model, total_steps, reset_timesteps=False):
    """progress_remaining SB3 will report on the first update of this run."""
    if reset_timesteps or total_steps <= 0:
        return 1.0
    return max(1.0 - model.num_timesteps / total_steps, 1e-6)


def set_lr_schedule(model, schedule):
    """Install `schedule` as the model's learning-rate schedule, for this run and for checkpoints."""
    model.learning_rate = schedule
    model.lr_schedule = schedule


class AdaptiveLRCallback(BaseCallback):
    """Multiplicative LR controller keyed on train/approx_kl.

    After each PPO update: approx_kl above 1.5 x target divides the level by `factor`, below
    target / 1.5 multiplies it; the dead band in between holds. The level is clamped to
    [lr_min, lr_max], changed at most every `adapt_every` updates and frozen for the first
    `freeze_updates` updates of a run (value refits after a resume make KL noisy). The base
    schedule's shape still applies on top (a linear base decays the adapted level), and the
    current level is stored on the model as `adaptive_lr` so checkpoints resume from it.
    """

    def __init__(self, level, progress_at_start=1.0, decay='constant', kl_target=0.012, lr_min=1e-5, lr_max=1e-4,
                 adapt_every=5, freeze_updates=20, factor=1.5, reset=False, verbose=1):
        super().__init__(verbose)
        self.progress_at_start = progress_at_start
        self.decay = decay
        self.kl_target = kl_target
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.adapt_every = max(int(adapt_every), 1)
        self.freeze_updates = int(freeze_updates)
        self.factor = factor
        self.reset = reset
        self.level = float(level)
        self.updates_seen = 0
        self.updates_since_change = 0

    def _clamp(self, level):
        return min(max(level, self.lr_min), self.lr_max)

    def _apply(self):
        self.level = self._clamp(self.level)
        self.model.adaptive_lr = self.level
        set_lr_schedule(self.model, anchored_schedule(self.level, self.progress_at_start, self.decay))

    def _on_training_start(self):
        saved = getattr(self.model, 'adaptive_lr', None)
        if saved is not None and not self.reset:
            self.level = float(saved)
            if self.verbose:
                print(f"Adaptive LR resumes at {self.level:.3g} from the checkpoint")
        self._apply()

    def _on_rollout_start(self):
        # Called after train(): the last update's approx_kl is still in the logger until the next dump.
        kl = self.logger.name_to_value.get('train/approx_kl')
        if kl is None:
            return
        kl = float(kl)
        self.updates_seen += 1
        self.updates_since_change += 1
        self.logger.record('train/adaptive_lr', self.level)
        if self.updates_seen <= self.freeze_updates or self.updates_since_change < self.adapt_every:
            return
        if kl > self.kl_target * self.factor:
            level = self.level / self.factor
        elif kl < self.kl_target / self.factor:
            level = self.level * self.factor
        else:
            return
        level = self._clamp(level)
        if level == self.level:
            return
        if self.verbose:
            print(f"Adaptive LR: approx_kl {kl:.4f} vs target {self.kl_target:.4f}, lr {self.level:.3g} -> {level:.3g}")
        self.level = level
        self.updates_since_change = 0
        self._apply()

    def _on_step(self):
        return True
