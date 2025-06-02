import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
from typing import Optional
from typing import Sequence

from utilities import weight_initialization
from utilities import get_multilayer_perceptron
from utilities import MyCovn1d


class StochasticPolicy(nn.Module):
    def __init__(
        self,
        input_dims: int,
        num_actions: int,
        hidden_units: Optional[Sequence[int]] = None,
        action_space=None,
        epsilon: float = 10e-6,
        log_sigma_max: float = 2,
        log_sigma_min: float = -20,
        num_type: float = 10,
        history_length=None
    ):
        super(StochasticPolicy, self).__init__()

        self.cnn = MyCovn1d(channels=num_type*6)
        self.history_length = history_length
        self.T_input_dim = int(input_dims / history_length)
        if hidden_units is None:
            hidden_units = [256, 256]

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        self.epsilon = epsilon
        self.log_sigma_max = log_sigma_max
        self.log_sigma_min = log_sigma_min

        self.num_type = num_type
        self.ffn_norm1 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm5 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm6 = nn.LayerNorm(int(num_type), eps=1e-6)

        units = [input_dims] + list(hidden_units)
        self.multilayer_perceptron = get_multilayer_perceptron(
            units, keep_last_relu=True)
        self.mean_linear = nn.Linear(units[-1], num_actions)
        self.log_std_linear = nn.Linear(units[-1], num_actions)

        if action_space is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0)

        self.apply(weight_initialization)

    def forward(self, x):  # todo maybe merge forward and evaluate + maybe add the "act" from openai
        state = []
        for i in range(self.history_length):
            state.append(
                x[:, (i) * self.T_input_dim: (i + 1)
                  * self.T_input_dim].unsqueeze(1)  # 在第一维度增加一个维度
            )

        x = torch.cat(state, 1)  # 调整顺序
        x = torch.cat(
            [
                self.ffn_norm1(x[:, :, : self.num_type]),
                self.ffn_norm2(x[:, :, self.num_type: 2*self.num_type]),
                self.ffn_norm3(
                    x[:, :, 2*self.num_type: 3*self.num_type]
                ),
                self.ffn_norm4(
                    x[:, :, 3*self.num_type: 4*self.num_type]
                ),
                self.ffn_norm5(
                    x[
                        :,
                        :,
                        4*self.num_type:5*self.num_type
                    ]
                ),
                self.ffn_norm6(x[:, :, -self.num_type:]),
            ],
            -1,
        )  # 取均值再合并

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)  # flattern 来喂进线性

        x = self.multilayer_perceptron(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std_clamped = torch.clamp(
            log_std, min=self.log_sigma_min, max=self.log_sigma_max)
        std = torch.exp(log_std_clamped)
        return mean, std

    def evaluate(self, state, deterministic: bool = False, with_log_probability: bool = True):
        mean, std = self.forward(state)
        distribution = Normal(mean, std)
        sample = distribution.rsample()

        if deterministic:
            # action = mean
            action = torch.tanh(mean)
        else:
            # todo when sac working, multiply by action_scale and add action_bias
            action = torch.tanh(sample)

        if with_log_probability:
            # Implementation that I originally implemented
            # the "_" are only here for now to debug the values and the shapes
            # log_probability_ = distribution.log_prob(sample) - torch.log((1 - action.pow(2)) + self.epsilon)
            # log_probability = log_probability_.sum(1, keepdim=True)

            # OPENAI Implementation
            # https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/sac/core.py#L59
            log_probability_ = distribution.log_prob(
                sample).sum(axis=-1, keepdim=True)
            log_probability__ = (
                2 * (np.log(2) - sample - F.softplus(-2 * sample))).sum(axis=1).unsqueeze(1)
            log_probability = log_probability_ - log_probability__
        else:
            log_probability = None

        return action, log_probability

    # todo need to replace in the agent code
    def act(self, observation, deterministic=False) -> np.array:
        with torch.no_grad():
            action, _ = self.evaluate(
                observation, deterministic=deterministic, with_log_probability=False)
            return action.cpu().numpy()[0]
