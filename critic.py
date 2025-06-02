import torch
import torch.nn as nn

from typing import List
from typing import Tuple
from typing import Optional
from typing import Sequence

from utilities import weight_initialization
from utilities import get_multilayer_perceptron
from utilities import MyCovn1d


class QNetwork(nn.Module):
    """Q Network module"""

    def __init__(self, input_dims: int, num_actions: int, hidden_units: List[int]):
        super(QNetwork, self).__init__()

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = hidden_units

        units = [input_dims + num_actions] + hidden_units + [1]
        self.multilayer_perceptron = get_multilayer_perceptron(
            units, keep_last_relu=False)

        self.apply(weight_initialization)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor = torch.cat([state, action], dim=1)
        q_value = self.multilayer_perceptron(tensor)
        return q_value


class TwinnedQNetworks(nn.Module):
    """Class containing two Q Networks"""

    def __init__(self, input_dims: int, num_actions: int, hidden_units: Optional[Sequence[int]] = None, num_type: float = 10, history_length=None):
        super(TwinnedQNetworks, self).__init__()
        self.history_length = history_length
        self.T_input_dim = int(input_dims / history_length)
        self.cnn = MyCovn1d(channels=num_type*6)
        self.num_type = num_type
        self.ffn_norm1 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm2 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm3 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm4 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm5 = nn.LayerNorm(int(num_type), eps=1e-6)
        self.ffn_norm6 = nn.LayerNorm(int(num_type), eps=1e-6)
        if hidden_units is None:
            hidden_units = [256, 256]

        self.input_dims = input_dims
        self.num_actions = num_actions
        self.hidden_units = list(hidden_units)

        self.q_network_1 = QNetwork(
            input_dims=input_dims, num_actions=num_actions, hidden_units=self.hidden_units)
        self.q_network_2 = QNetwork(
            input_dims=input_dims, num_actions=num_actions, hidden_units=self.hidden_units)

    def forward(self, x, action) -> Tuple[torch.Tensor, torch.Tensor]:
        state = []
        for i in range(self.history_length):
            state.append(
                x[:, (i) * self.T_input_dim: (i + 1)
                  * self.T_input_dim].unsqueeze(1)  
            )

        x = torch.cat(state, 1)  
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
        # 之前的是每个时间长度算一个
        x = torch.flatten(x, start_dim=1)  

        q_1 = self.q_network_1(x, action)
        q_2 = self.q_network_2(x, action)
        return q_1, q_2
