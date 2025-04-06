import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).view(-1))

    def forward(self, input):
        B = input.size(0)
        x = input.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(B, self.k, self.k)
        return torch.bmm(input, x)


class PIFE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PIFE, self).__init__()
        self.input_transform = TNet(k=input_dim)

        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x_transformed = self.input_transform(x)  # (B, N, m)
        x_transformed = x_transformed.transpose(2, 1)
        x_features = self.mlp(x_transformed)
        global_feature = torch.max(x_features, dim=2)[0]
        return global_feature

class MultiPIFE(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(MultiPIFE, self).__init__()
        self.pifes = nn.ModuleList([
            PIFE(input_dim=dim, output_dim=output_dim) for dim in input_dims
        ])

    def forward(self, inputs):
        pife_outputs = []
        for i, pife in enumerate(self.pifes):
            out = pife(inputs[i])
            pife_outputs.append(out)
        return torch.cat(pife_outputs, dim=1)

class PIFEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, pife_out_dim=64):
        self.input_dims = []
        self.pife_out_dim = pife_out_dim
        for space in obs_space.spaces.values():
            self.input_dims.append(space.shape[-1])
        self.num_entity_types = len(self.input_dims)

        total_output_dim = self.num_entity_types * self.pife_out_dim

        super().__init__(obs_space, features_dim=total_output_dim)

        self.multi_pife = MultiPIFE(input_dims=self.input_dims, output_dim=self.pife_out_dim)

    def forward(self, obs):
        inputs = []
        for key in sorted(obs.keys()):
            tensor = obs[key]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            inputs.append(tensor)
        return self.multi_pife(inputs)

class CombinedPIFEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, pife_out_dim=64):
        super().__init__(obs_space, features_dim=1)

        pife_keys = [k for k in obs_space.spaces.keys() if k.startswith("pife_")]
        flat_keys = [k for k in obs_space.spaces.keys() if not k.startswith("pife_")]

        self.pife_keys = sorted(pife_keys)
        self.flat_keys = sorted(flat_keys)

        pife_space = spaces.Dict({k: obs_space.spaces[k] for k in self.pife_keys})
        flat_space = spaces.Dict({k: obs_space.spaces[k] for k in self.flat_keys})

        self.pife_extractor = PIFEFeatureExtractor(pife_space, pife_out_dim=pife_out_dim)
        self.flat_extractor = nn.Flatten()

        total_features_dim = self.pife_extractor.features_dim + get_flattened_obs_dim(flat_space)
        self._features_dim = total_features_dim


    def forward(self, obs):
        pife_obs = {k: obs[k] for k in self.pife_keys}
        flat_obs = [obs[k].view(obs[k].size(0), -1) for k in self.flat_keys]

        pife_out = self.pife_extractor(pife_obs)
        flat_out = torch.cat(flat_obs, dim=1)

        return torch.cat([pife_out, flat_out], dim=1)
