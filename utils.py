import os
import datetime
import numpy as np
import random
import torch


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def check_existence(path_of_file: str):
    """
    根据文件名或者文件夹检查文件/文件夹是否存在
    """
    path_of_file = os.path.abspath(path_of_file)
    return os.path.exists(path_of_file)


def rescale_action(action, env):
    """
    对神经网络输出的动作输出到环境需要的范围
    """
    action_space = env.action_space

    caching_offloading_low = np.zeros((20,))
    caching_offloading_high = np.ones((20,))
    low = np.concatenate(
        (caching_offloading_low, action_space["uplink"].low, action_space["downlink"].low, action_space["frequency"].low))
    high = np.concatenate(
        (caching_offloading_high, action_space["uplink"].high, action_space["downlink"].high, action_space["frequency"].high))
    action_range = [low, high]

    return action * (action_range[1] - action_range[0]) / 2.0 + (action_range[1] + action_range[0]) / 2.0



def normal_action(action, env):
    """
    对sample出的动作转化到[-1,1]
    """
    action_space = env.action_space

    caching_offloading_low = np.zeros((20,))
    caching_offloading_high = np.ones((20,))
    low = np.concatenate(
        (caching_offloading_low, action_space["uplink"].low, action_space["downlink"].low, action_space["frequency"].low))
    high = np.concatenate(
        (caching_offloading_high, action_space["uplink"].high, action_space["downlink"].high, action_space["frequency"].high))
    action_range = [low, high]
    return (action - ((action_range[1] + action_range[0]) / 2.0)) / ((action_range[1] - action_range[0]) / 2.0)


def rescale_action_cvx(action, env):

    caching_offloading_low = np.zeros((2*env.num_service_type,))
    caching_offloading_high = np.ones((2*env.num_service_type,))

    action_range = [caching_offloading_low, caching_offloading_high]

    return action * (action_range[1] - action_range[0]) / 2.0 + (action_range[1] + action_range[0]) / 2.0



def normal_action_cvx(action, env):

    caching_offloading_low = np.zeros((2*env.num_service_type,))
    caching_offloading_high = np.ones((2*env.num_service_type,))

    action_range = [caching_offloading_low, caching_offloading_high]
    return (action - ((action_range[1] + action_range[0]) / 2.0)) / ((action_range[1] - action_range[0]) / 2.0)


class Logger(object):
    """
    根据文件名来将日志写进文件
    可以选择输出到屏幕还是写到文件
    判断文件存不存在不存在新建文件
    """

    def __init__(self, log_path=None):
        if check_existence("logs"):
            pass
        else:
            os.mkdir("logs")  # 生成文件夹
        if log_path is not None:
            self.log_path = log_path
        else:
            self.log_path = "{}_log.txt".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.f = open("logs/{}".format(self.log_path), "a+",
                      encoding="utf-8")  # 默认是追加，不存在则自动创建

    def log2file(self, msg: str):
        """
        输出信息到指定的文件
        """
        print(msg, file=self.f,
              flush=True)  # 可以看到text.txt文件这时还是为空，只有f.close()后才将内容写进文件中。flush可以马上写到文件中

    def log2screen(self, msg: str):
        """
        输出信息到屏幕中
        """
        print(msg, file=None, flush=True)

    def log(self, msg, toScreen=False):
        """
        输出信息到文件和屏幕
        """
        self.log2file(msg)
        if toScreen:
            self.log2screen(msg)

    def close(self):
        self.f.close()


# f = open("out.txt", "a+")
# print("23", file=f)
# print("123", file=None)  # print("123")
# # print("123")
# f.close()

# log = Logger()

# log.log("321")
