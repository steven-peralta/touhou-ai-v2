import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class PIFE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(PIFE, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        present = x[:, :, 0] > 0.5
        x_features = self.mlp(x)
        x_features = x_features.masked_fill(~present.unsqueeze(-1), 0.0)
        global_feature = torch.max(x_features, dim=1)[0]
        return global_feature

class MultiPIFE(nn.Module):
    def __init__(self, input_dims, output_dim, hidden_dim=256):
        super(MultiPIFE, self).__init__()
        self.pifes = nn.ModuleList([
            PIFE(input_dim=dim, output_dim=output_dim, hidden_dim=hidden_dim) for dim in input_dims
        ])

    def forward(self, inputs):
        pife_outputs = []
        for i, pife in enumerate(self.pifes):
            out = pife(inputs[i])
            pife_outputs.append(out)
        return torch.cat(pife_outputs, dim=1)

class PIFEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, pife_out_dim=128, pife_hidden_dim=256):
        self.input_dims = []
        self.pife_out_dim = pife_out_dim
        for space in obs_space.spaces.values():
            self.input_dims.append(space.shape[-1])
        self.num_entity_types = len(self.input_dims)

        total_output_dim = self.num_entity_types * self.pife_out_dim

        super().__init__(obs_space, features_dim=total_output_dim)

        self.multi_pife = MultiPIFE(input_dims=self.input_dims, output_dim=self.pife_out_dim, hidden_dim=pife_hidden_dim)

    def forward(self, obs):
        inputs = []
        for key in sorted(obs.keys()):
            tensor = obs[key]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            inputs.append(tensor)
        return self.multi_pife(inputs)

class CombinedPIFEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, pife_out_dim=128, pife_hidden_dim=256):
        super().__init__(obs_space, features_dim=1)

        pife_keys = [k for k in obs_space.spaces.keys() if k.startswith("pife_")]
        flat_keys = [k for k in obs_space.spaces.keys() if not k.startswith("pife_")]

        self.pife_keys = sorted(pife_keys)
        self.flat_keys = sorted(flat_keys)

        pife_space = spaces.Dict({k: obs_space.spaces[k] for k in self.pife_keys})
        flat_space = spaces.Dict({k: obs_space.spaces[k] for k in self.flat_keys})

        self.pife_extractor = PIFEFeatureExtractor(pife_space, pife_out_dim=pife_out_dim, pife_hidden_dim=pife_hidden_dim)
        self.flat_extractor = nn.Flatten()

        total_features_dim = self.pife_extractor.features_dim + get_flattened_obs_dim(flat_space)
        self._features_dim = total_features_dim


    def forward(self, obs):
        pife_obs = {k: obs[k] for k in self.pife_keys}
        flat_obs = [obs[k].view(obs[k].size(0), -1) for k in self.flat_keys]

        pife_out = self.pife_extractor(pife_obs)
        flat_out = torch.cat(flat_obs, dim=1)

        return torch.cat([pife_out, flat_out], dim=1)
