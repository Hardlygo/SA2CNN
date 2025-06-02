import datetime
from pathlib import Path
from contextlib import contextmanager
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from typing import Union
from typing import Optional


def get_run_name(*arguments, **keyword_arguments) -> str:
    name = f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    if arguments:
        for argument in arguments:
            name += f'_{str(argument)}'
    if keyword_arguments:
        for argument_name, argument_value in keyword_arguments.items():
            name += f'_{argument_name}{str(argument_value)}'
    return name


@contextmanager
def eval_mode(net: nn.Module):
    """Temporarily switch to evaluation mode."""
    originally_training = net.training
    try:
        net.eval()
        yield net
    finally:
        if originally_training:
            net.train()


def get_device(overwrite: Optional[bool] = None, device: str = 'cuda:0', fallback_device: str = 'cpu') -> torch.device:
    use_cuda = torch.cuda.is_available() if overwrite is None else overwrite
    return torch.device(device if use_cuda else fallback_device)


def update_network_parameters(source: nn.Module, target: nn.Module, tau: float) -> None:
    for target_parameters, source_parameters in zip(target.parameters(), source.parameters()):
        target_parameters.data.copy_(
            target_parameters.data * (1.0 - tau) + source_parameters.data * tau)


def save_model(model: nn.Module, file: Union[str, Path]) -> None:
    try:
        torch.save(model.state_dict(), file)
    except FileNotFoundError:
        print('Unable to save the models')


def load_model(model: nn.Module, file: Union[str, Path], device) -> None:
    try:
        model.load_state_dict(torch.load(file))
        model.to(device)
    except FileNotFoundError:
        print('Unable to load the models')


def save_to_writer(writer, tag_to_scalar_value: dict, step: int) -> None:
    for tag, scalar_value in tag_to_scalar_value.items():
        writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=step)


def filter_info(dictionary: dict) -> dict:
    filtered_dictionary = {}
    for key, value in dictionary.items():
        if '/' in key and isinstance(value, (int, float)):
            filtered_dictionary[key] = value
    return filtered_dictionary


def weight_initialization(module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


def get_multilayer_perceptron(unit_list: List[int], keep_last_relu: bool = False) -> nn.Sequential:
    module_list = []
    for in_features, out_features in zip(unit_list, unit_list[1:]):
        module_list.append(nn.Linear(in_features, out_features))
        module_list.append(nn.ReLU())
    if keep_last_relu:
        return nn.Sequential(*module_list)
    else:
        return nn.Sequential(*module_list[:-1])


def get_timedelta_formatted(td):
    hours, rem = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # pe是二维 max_len*d_model
        position = torch.arange(
            0, max_len, dtype=torch.float).unsqueeze(1)  # max_len*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # max_len/2
        pe[:, 0::2] = torch.sin(position * div_term)  # 对应位置相乘
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # max_len*1*d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CNNBlock(torch.nn.Module):

    def __init__(self, n_filters=40, embedding_dim=10, dropout=0.5):
        super(CNNBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(1, embedding_dim))
        # self.conv_1 = nn.Conv2d(in_channels = 1,
        #                         out_channels = n_filters,
        #                         kernel_size = (embedding_dim, 1))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x= [batch size, sent len, emb dim]
        x = x.unsqueeze(1)
        # x= [batch size, 1, sent len, emb dim]
        conved_0 = F.relu(self.conv_0(x).squeeze(3))
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        # pooled_n = [batch size, n_filters]
        return self.dropout(pooled_0)


class CNN1d(torch.nn.Module):
    def __init__(self,  embedding_dim=10, n_filters=10, filter_sizes=[1, 3],
                 dropout=0.5):
        super(CNN1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs, padding=(fs-1)//2)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, state):
        # x= [batch size, sent len, emb dim]=[N,4,10]
        x = state.permute(0, 2, 1)
        # #embedded = [batch size, emb dim, sent len]
        conved = [F.relu(conv(x)) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        # pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        # cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        y = conved[1].permute(0, 2, 1)

        return state+y


class MyCovn1d(torch.nn.Module):
    # conv1 conv2 maxpool dropout
    def __init__(self, channels, dropout=0.5):
        super(MyCovn1d, self).__init__()
        self.convs = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channels, out_channels=256,
                            kernel_size=3, padding=1),  # inchannel代表输入单个T的长度 之前是60 120 现在改为72 144
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=256, out_channels=channels,
                            kernel_size=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2, 2)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, state):
        # state=[N,1,40][channel,len]
        state = self.convs(state)
        # state=[N,10,20][channel,len]
        return self.dropout(state)
