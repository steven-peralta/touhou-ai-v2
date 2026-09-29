import glob
import os

import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib.common.recurrent.type_aliases import RNNStates

HIT_REWARD_THRESHOLD = -1.5
STEPS_BEFORE_HIT = 30
STEPS_AFTER_HIT = 32
MIN_EPISODES_FOR_HOLDOUT = 10
SEQUENCE_LENGTH = 32


class DemoDataset:
    def __init__(self, demo_dir, frame_stack_size, action_repeat, gamma, val_fraction=0.1):
        paths = sorted(glob.glob(os.path.join(demo_dir, '*.npz')))
        if not paths:
            raise ValueError(f"No demonstrations found in {demo_dir}")

        obs, actions, returns, starts, imitate, train_idx, val_idx = {}, [], [], [], [], [], []
        n_val_episodes = max(1, int(len(paths) * val_fraction)) if len(paths) >= MIN_EPISODES_FOR_HOLDOUT else 0
        offset = 0
        for episode, path in enumerate(paths):
            with np.load(path) as data:
                episode_obs = {k[len('obs/'):]: data[k] for k in data.files if k.startswith('obs/')}
                episode_actions = data['actions']
                rewards = data['rewards']
                demo_action_repeat = int(data['action_repeat']) if 'action_repeat' in data.files else 1
                hits = data['hits'] if 'hits' in data.files else rewards < HIT_REWARD_THRESHOLD

            if demo_action_repeat != action_repeat:
                raise ValueError(f"{path} was recorded with action repeat {demo_action_repeat}, training uses {action_repeat}")

            length = len(episode_actions)
            episode_returns = np.zeros(length, dtype=np.float32)
            running = 0.0
            for t in range(length - 1, -1, -1):
                running = rewards[t] + gamma * running
                episode_returns[t] = running

            for key, value in episode_obs.items():
                obs.setdefault(key, []).append(value)
            actions.append(episode_actions)
            returns.append(episode_returns)
            starts.append(np.full(length, offset, dtype=np.int64))

            episode_imitate = np.ones(length, dtype=bool)
            for t in np.flatnonzero(hits):
                episode_imitate[max(0, t - STEPS_BEFORE_HIT):t + STEPS_AFTER_HIT + 1] = False
            imitate.append(episode_imitate)

            indices = np.arange(offset, offset + length)
            if n_val_episodes:
                (val_idx if episode >= len(paths) - n_val_episodes else train_idx).append(indices)
            else:
                n_val = int(length * val_fraction)
                train_idx.append(indices[:length - n_val])
                val_idx.append(indices[length - n_val:])
            offset += length

        self.obs = {k: np.concatenate(v) for k, v in obs.items()}
        self.actions = np.concatenate(actions)
        self.returns = np.concatenate(returns)
        self.starts = np.concatenate(starts)
        self.imitate = np.concatenate(imitate)
        self.train_idx = np.concatenate(train_idx)
        self.val_idx = np.concatenate(val_idx)
        self.frame_stack_size = frame_stack_size
        self.n_episodes = len(paths)
        self.ends = np.zeros(len(self.actions), dtype=np.int64)
        for start in np.unique(self.starts):
            mask = self.starts == start
            self.ends[mask] = start + mask.sum()
        self.train_idx = self.train_idx[self.train_idx + SEQUENCE_LENGTH <= self.ends[self.train_idx]]
        self.val_idx = self.val_idx[self.val_idx + SEQUENCE_LENGTH <= self.ends[self.val_idx]]

    def __len__(self):
        return len(self.actions)

    def check_spaces(self, observation_space, action_space):
        if set(self.obs) != set(observation_space.spaces):
            raise ValueError(f"Demo observation keys {sorted(self.obs)} do not match env keys {sorted(observation_space.spaces)}")
        for key, space in observation_space.spaces.items():
            shape = self.obs[key].shape[1:-1] + (self.obs[key].shape[-1] * self.frame_stack_size,)
            if shape != space.shape:
                raise ValueError(f"Demo observation '{key}' has stacked shape {shape}, env expects {space.shape}")
        if self.actions.shape[1:] != action_space.shape or np.any(self.actions.max(axis=0) >= action_space.nvec):
            raise ValueError(f"Demo actions do not fit the env action space {action_space}")

    def sequence_batch(self, window_starts, device):
        indices = (window_starts[:, None] + np.arange(SEQUENCE_LENGTH)[None, :]).reshape(-1)
        obs, actions, returns, imitate = self.batch(indices, device)
        episode_starts = th.zeros(len(indices), device=device)
        episode_starts[::SEQUENCE_LENGTH] = 1.0
        return obs, actions, returns, imitate, episode_starts

    def batch(self, indices, device):
        starts = self.starts[indices]
        obs = {}
        for key, value in self.obs.items():
            frames = []
            for back in range(self.frame_stack_size - 1, -1, -1):
                frame_idx = indices - back
                frame = value[np.maximum(frame_idx, starts)].copy()
                frame[frame_idx < starts] = 0
                frames.append(frame)
            obs[key] = th.as_tensor(np.concatenate(frames, axis=-1), device=device)
        actions = th.as_tensor(self.actions[indices], device=device)
        returns = th.as_tensor(self.returns[indices], device=device)
        imitate = th.as_tensor(self.imitate[indices], device=device)
        return obs, actions, returns, imitate


def zero_states(policy, n_seq, device):
    shape = (policy.lstm_actor.num_layers, n_seq, policy.lstm_actor.hidden_size)
    zeros = lambda: th.zeros(shape, device=device)
    return RNNStates((zeros(), zeros()), (zeros(), zeros()))


def evaluate_sequences(policy, obs, actions, episode_starts, device):
    n_seq = int(episode_starts.sum().item())
    states = zero_states(policy, n_seq, device)
    values, log_prob, entropy = policy.evaluate_actions(obs, actions, states, episode_starts)
    features = policy.extract_features(obs)
    if isinstance(features, tuple):
        features = features[0]
    latent_pi, _ = policy._process_sequence(features, states.pi, episode_starts, policy.lstm_actor)
    latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
    predicted = policy._get_action_dist_from_latent(latent_pi).mode()
    return values, log_prob, entropy, predicted


def evaluate(policy, dataset, indices, batch_size, device):
    policy.set_training_mode(False)
    nll, correct, value_error, n_imitate = 0.0, 0.0, 0.0, 0
    n_windows = max(batch_size // SEQUENCE_LENGTH, 1)
    with th.no_grad():
        for i in range(0, len(indices), n_windows):
            obs, actions, returns, imitate, episode_starts = dataset.sequence_batch(indices[i:i + n_windows], device)
            values, log_prob, _, predicted = evaluate_sequences(policy, obs, actions, episode_starts, device)
            nll -= log_prob[imitate].sum().item()
            correct += (predicted == actions)[imitate].float().sum(dim=0).cpu().numpy()
            value_error += F.mse_loss(values.flatten(), returns, reduction='sum').item()
            n_imitate += imitate.sum().item()
    n_imitate = max(n_imitate, 1)
    return nll / n_imitate, correct / n_imitate, value_error / max(len(indices) * SEQUENCE_LENGTH, 1)


def pretrain(
        model,
        demo_dir,
        frame_stack_size,
        action_repeat,
        epochs=10,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.001,
        vf_coef=0.5,
        fit_value=True,
):
    policy = model.policy
    device = model.device

    dataset = DemoDataset(demo_dir, frame_stack_size, action_repeat, model.gamma)
    dataset.check_spaces(model.observation_space, model.action_space)
    print(f"Pretraining on {dataset.n_episodes} episodes, {len(dataset.train_idx)} train / {len(dataset.val_idx)} val windows "
          f"of {SEQUENCE_LENGTH} steps, {1 - dataset.imitate.mean():.1%} of steps excluded from imitation around hits")
    n_windows = max(batch_size // SEQUENCE_LENGTH, 1)
    if not fit_value:
        # Demo rewards were computed under whichever reward the recorder ran with; skip the
        # value-head fit when that no longer matches the training reward.
        print("Skipping the value-head fit; only imitating demo actions")
        vf_coef = 0.0

    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng()
    metrics = {}

    for epoch in range(epochs):
        policy.set_training_mode(True)
        indices = rng.permutation(dataset.train_idx)
        for i in range(0, len(indices), n_windows):
            batch_idx = indices[i:i + n_windows]
            if len(batch_idx) < 2:
                continue
            obs, actions, returns, imitate, episode_starts = dataset.sequence_batch(batch_idx, device)
            values, log_prob, entropy = policy.evaluate_actions(obs, actions, zero_states(policy, len(batch_idx), device), episode_starts)
            weight = imitate.float()
            policy_loss = -(log_prob * weight).sum() / weight.sum().clamp(min=1.0)
            loss = policy_loss - ent_coef * entropy.mean() + vf_coef * F.mse_loss(values.flatten(), returns)

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), model.max_grad_norm)
            optimizer.step()

        eval_idx = dataset.val_idx if len(dataset.val_idx) else dataset.train_idx
        nll, accuracy, value_error = evaluate(policy, dataset, eval_idx, batch_size, device)
        metrics = {
            'pretrain/val_nll': nll,
            'pretrain/val_value_mse': value_error,
            'pretrain/val_acc_direction': accuracy[0],
            'pretrain/val_acc_shoot': accuracy[1],
            'pretrain/val_acc_focus': accuracy[2],
        }
        print(f"[pretrain {epoch + 1}/{epochs}] nll={nll:.4f} value_mse={value_error:.4f} "
              f"acc direction={accuracy[0]:.3f} shoot={accuracy[1]:.3f} focus={accuracy[2]:.3f}")

    policy.set_training_mode(False)
    return metrics
