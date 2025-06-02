import torch
import datetime
import numpy as np
from pathlib import Path
from itertools import count
from typing import Optional
from typing import Sequence

from tensorboardX import SummaryWriter

from agent import Agent

from utilities import filter_info
from utilities import get_run_name
from utilities import save_to_writer

from utilities import get_timedelta_formatted
from environment_his_state_cvx import Environment

from utils import seed_torch,  normal_action_cvx,  rescale_action_cvx


def train(
    batch_size: int = 64,
    memory_size: int = 1000000,
    learning_rate: float = 3e-4,
    alpha_learning_rate: float = 3e-4,
    alpha: float = 0.05,
    gamma: float = 0.99,
    tau: float = 0.005,
    num_steps: int = 2000000,
    hidden_units: Optional[Sequence[int]] = [256, 256],
    load_models: bool = False,
    saving_frequency: int = 20,
    start_step: int = 1000,
    seed: int = 0,
    updates_per_step: int = 1,
    directory: str = "models/",
    sub_directory: str = None,
    num_task: int = 40,
    ap_storage_capacity: int = 300,
    spectral_efficiency: int = 3,
    bandwith: int = 20,  # 上下带宽
    ap_computing_capacity: int = 31,  # ap计算能力
    cloud_computing_rate: int = 5,  # C计算能力
    ap2c_rate: int = 8,  # ap到c的计算能力
    **kwargs,
):

    num_service_type = int(16) 
    history_length = 3
    np.random.seed(seed)
    seed_torch(seed)

    env = Environment(
        num_service_type=num_service_type,
        num_task=num_task,
        ap_storage_capacity=ap_storage_capacity,
        spectral_efficiency=spectral_efficiency,
        bandwith=bandwith,
        ap_computing_capacity=ap_computing_capacity,
        cloud_computing_rate=cloud_computing_rate,
        ap2c_rate=ap2c_rate,
        history_length=history_length,
        seed=seed
    )
    env.seed(seed)
    env.action_space.seed(seed)

    num_inputs = num_service_type * 6*history_length 
    num_actions = num_service_type * 2  

    writer = SummaryWriter(
        "runs/{}_SAC_A_{}_{}".format(
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "SERVICE_OFFLOAD_BANDWITH_RESOURCE_ALLOCATE",
            "Gaussian",
        )
    )
    # 按运行时间来分子文件夹
    if sub_directory is None:
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
        curr_time = curr_time + "/"
    else:
        curr_time = sub_directory

    model_directory = directory + curr_time  # 加上时间文件

    agent = Agent(
        observation_shape=num_inputs,
        num_actions=num_actions,
        action_space=env.action_space,
        alpha=alpha,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        hidden_units=hidden_units,
        batch_size=batch_size,
        memory_size=memory_size,
        checkpoint_directory=model_directory,
        load_models=load_models,
        target_update_interval=1,
        alpha_learning_rate=alpha_learning_rate,
        num_type=num_service_type,
        history_length=history_length
    )

    start_training_time = datetime.datetime.now()

    updates = 0
    global_step = 0
    score_history = []

    for episode in count():
        score = 0
        done = False
        episode_step = 0
        t_cost = 0
        e_cost = 0
        sum_usage_R = 0
        sum_usage_up = 0
        sum_usage_down = 0
        sum_usage_CPU = 0

        observation = env.reset()

        while not done:
            if start_step > global_step:
                action_ = env.sample()  
                action = normal_action_cvx(action_, env)
            else:
                action = agent.choose_action(observation)
                action_ = rescale_action_cvx(action, env)

            new_observation, reward, done, t_cost_per_step, e_cost_per_step, usage_R, usage_up, usage_down, usage_CPU = env.step(
                action_, observation)
            agent.remember(observation, action, reward, new_observation, done)


            sum_usage_R += usage_R
            sum_usage_up += usage_up
            sum_usage_down += usage_down
            sum_usage_CPU += usage_CPU

            score += reward
            t_cost += t_cost_per_step
            e_cost += e_cost_per_step
            global_step += 1
            episode_step += 1
            observation = new_observation

            if agent.memory.memory_counter >= batch_size:
                for update in range(updates_per_step):
                    tensorboard_logs = agent.learn(updates)
                    save_to_writer(writer, tensorboard_logs, updates)
                    updates += 1

        score_history.append(score)
        average_score = np.mean(score_history[-100:])
        time_delta = get_timedelta_formatted(
            datetime.datetime.now() - start_training_time)

        tensorboard_logs = {
            "train/reward": score,
            "train/time": t_cost,
            "train/avg_time_per_task": t_cost/(env._max_episode_steps),
            "train/energy": e_cost,
            "train/avg_energy_per_task": e_cost/(env._max_episode_steps)
            # **filter_info(info),
        }
        save_to_writer(writer, tensorboard_logs, episode)

        #!计算平均利用率
        avg_usage_logs = {
            "avg_usage/storage": sum_usage_R/env._max_episode_steps,
            "avg_usage/uplink": sum_usage_up/env._max_episode_steps,
            "avg_usage/downlink": sum_usage_down/env._max_episode_steps,
            "avg_usage/CPU": sum_usage_CPU/env._max_episode_steps
            # **filter_info(info),
        }
        save_to_writer(writer, avg_usage_logs, episode)

        if episode % saving_frequency == 0:
            last_save_episode = episode
            agent.save_models()

        if global_step > num_steps:
            break
    writer.close()
    print("train end!!")
