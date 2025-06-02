# 这个文件主要是换一个奖励函数
# r(t)=avg_d(t)+1.5*percent_usage_resource
import numpy as np
from gymnasium import spaces
import random
import math

from covxpy import cvxpy_loss
from other import *

from distance_source_sampler import DataSampler
from chanel_gain_generator import DynamicDistanceChannelGenerator
import datetime


class Environment(object):
    """
    带有服务缓存、带宽分配、资源分配的MEC环境
    """

    def __init__(
        self,
        num_service_type=10,
        num_task=40,
        ap_storage_capacity=300,  # ap存储(GB)
        spectral_efficiency=3,  # 信道频谱效率bit/s/Hz
        bandwith=20,  # 上下带宽MHz
        ap_computing_capacity=20,  # ap计算能力 默认=20GHz
        cloud_computing_rate=5,  # C计算能力5GHz
        ap2c_rate=15,  # ap到c的计算能力以前是8Mb/s
        history_length=3,
        seed=2023
    ):
        super(Environment, self).__init__()

        self.num_service_type = num_service_type  # 默认是10个服务类型

        self._max_episode_steps = 201  # 一个episode 100步
        self.current_steps = 0
        # QoE-Aware Decentralized Task Offloading andResource Allocation for End-Edge-CloudSystems:A Game-Theoretical Approach
        self.trans_power = 1e-3*pow(10., 30./10.)
        self.num_task = num_task

        # *环境相关
        self.bandwith_up = bandwith  # 20MHZ 升到25？
        self.bandwidth_down = bandwith
        self.spectral_efficiency = spectral_efficiency  # bit/s/HZ
        self.ap_storage_capacity = ap_storage_capacity  # 300G
        self.ap_computing_capacity = ap_computing_capacity * \
            1000  # 20x1e9HZ=20ghz 变30G或35G或40G
        self.cloud_computing_rate = cloud_computing_rate*1000  # 5000MHz 5x10^9HZ=5GHz
        self.ap2c_rate = ap2c_rate  # 15  # 15Mbps化为bps #变10m或8m

        self.history_length = history_length

        self.user_distance_sampler = DataSampler(
            target_column='distance_km',
            shuffle=True
        )
        self.channel_generator = DynamicDistanceChannelGenerator(
            num_users=self.num_task,
            time_slots=self._max_episode_steps+history_length+1,
            seed=seed
        )
        # gen_service_space()  #[62. 56. 51. 65. 62. 77. 44. 75. 79. 69.] 生成每个类型所占用空间[63.92, 64.16, 72.12, 58.72, 77.99, 47.74 ,72.65, 41,   64.37, 80.33]
        self.service_space_arry = np.array(
            [62.0, 56.0, 51.0, 65.0, 62.0, 77.0, 44.0, 75.0, 79.0, 69.0, 52.0, 47.0, 58.0, 49.0, 46.0, 61.0])
# 58. 46. 49. 61.
        # ?字典型的动作空间
        self.action_space = spaces.Dict(
            {'offloading': spaces.MultiBinary(self.num_service_type), 'caching': spaces.MultiBinary(self.num_service_type)})

        # (input_size_service,output_size_service,computing_service_current,num_times_current,caching)
        self.state_t = np.zeros((self.num_service_type*6*self.history_length,))
        # 计算需求取决于输入和服务类型
        self.cycle_proportion = np.ones(
            (self.num_service_type,))  # *1500  # cycle 不同类型的计算需求比例
        # self.cycle_proportion=np.array([330,530,730,930,1130,1330,1530,1730,1930,2130]) #byte/cycle 不同类型的计算需求比例

    def step(self, action, state):
        """
        一个想法：
       最小化时延，且最大化ES资源利用率
        r=-d_t+α*usage_R+β*(usage_uplink+usage_downlink)+γ*usage_frequency
        """
        '''
        1.分离action，抽离各个部分的动作
        2.计算reward
        3.产生newstate
        '''
        action = np.round(action)  # <=0.5的会等于0,>0.5会等于1
        # *每个部分决策
        # ? action (caching,offloading,uplink,downlink,frequency)
        # 产生出来了，但是对当前时隙的奖励没有贡献啊
        caching_tplus1 = action[: self.num_service_type * 1][:]
        offloading = action[self.num_service_type *
                            1:self.num_service_type * 2][:]

        # ? state_t  (input_size_service,output_size_service,computing_service_current,num_times_current,caching)
        s_t = state[-self.num_service_type * 6:][:]  # 前60个元素代表当前状态
        input_size_service = s_t[:self.num_service_type]  # 上传数据量
        output_size_service = s_t[self.num_service_type *
                                  1:self.num_service_type*2]  # 结果数据量
        computing_service_current = s_t[self.num_service_type *
                                        2:self.num_service_type*3]  # 计算量
        efficiency = s_t[-self.num_service_type*2:-self.num_service_type:]
        caching_t = s_t[-self.num_service_type:]

        # print('input_size_service', input_size_service)
        # print('output_size_service', output_size_service)
        # print('computing_service_current', computing_service_current)

        # print('efficiency', efficiency)

        # 根据约束条件o_t(offloading)<=a_t(caching)处理offloading
        # ? 0代表不缓存，0代表不卸载 卸载范围是[0,1]
        # for i in range(len(offloading)): #?没有缓存的服务，无法支持卸载
        #     if caching_t[i]==0: #不缓存的都会不卸载，都会云上执行
        #         offloading[i]=0

        for i in range(len(offloading)):  # 不存在了只缓存，不卸载
            if offloading[i] > caching_t[i]:
                offloading[i] = 0

        # 缓存决策的缓存利用率计算
        left_space_R_flag = self.ap_storage_capacity  # 用过了一次，复位
        # 计算当前caching decision的缓存空间利用率，并调整cahing decision
        for i in range(self.num_service_type):
            #!能缓存尽量缓存，无法缓存了置决策为0
            # 缓存服务，减去空间
            if caching_tplus1[i] == 1 and left_space_R_flag >= self.service_space_arry[i]:
                # 这保证了left_space_R>=0
                left_space_R_flag -= self.service_space_arry[i]
                # caching_tplus1[i]=1
            else:
                caching_tplus1[i] = 0

        #!这是协调后的决策
        # print(caching_t, "caching")
        # print(offloading, "offloading")

        # ? left_space_R_flag肯定为正的，因为是能减才减
        usage_R = 1-(left_space_R_flag/self.ap_storage_capacity)
        loss1 = usage_R   # 可以出一个，缓存决策的准确率，和利用率

        # ? 改下具有梯度的奖励函数，收敛会不会更稳定？不会！
        # install_plan_usage=(caching_t*self.service_space_arry).sum()
        # penalization=-1*(install_plan_usage-self.ap_storage_capacity)/self.ap_storage_capacity #?超出约束，超出越多惩罚越大
        # positive_rewards=(install_plan_usage/self.ap_storage_capacity) #?没有超出约束，是小于1的，越大越好
        # loss1=positive_rewards if satify_space_constaint else penalization
        print("efficiency:", efficiency)
        # ?放到凸优化里面后，没有了三个资源约束，只考虑缓存容量约束和卸载与缓存约束
        cost, loss2, loss3, loss4, delay, energy, _, _, _ = cvxpy_loss(caching_t, offloading, input_size_service, output_size_service, computing_service_current, variable_num=self.num_service_type,
                                                                       ig=efficiency, upper_bandwidth=self.bandwith_up, upper_cpu=self.ap_computing_capacity, e2c_rate=self.ap2c_rate,
                                                                       cloud_computing_rate=self.cloud_computing_rate, cycle_proportion=self.cycle_proportion, trans_power=self.trans_power)
        reward = (
            -(cost/self.num_task)
            # 04-27=1.55+ 0.3*(loss1+loss4)+0.3*(loss2+loss3)
            + 0.5*(loss1+loss4)+0.1*(loss2+loss3)
        )

        # reward = (
        #     math.exp((2-(cost/self.num_task))/2.0) -
        #     6*((1-loss1)+(1-loss4)+(1-loss2)+(1-loss3))  # 04-27=1.55
        # )

        # * 是否可以reward=reward/num_service_type 求每个类型的均值reward
        print("dalay:", delay)
        print("energy:", energy)

        print(loss1, loss2, loss3, loss4)
        # print("reward:", reward)

        self.current_steps += 1
        print("current_steps", self.current_steps)
        print("max_steps", self._max_episode_steps)
        done = True if self.current_steps == (
            self._max_episode_steps+self.history_length) else False
        # ?generate new state
        # ? state  (input_size_service,output_size_service,computing_service_current,num_times_current,caching)
        # ?state=[st,st-1,st-2]
        state_t = np.concatenate((self.gen_new_state(), caching_tplus1[:]))
        state_t_1 = s_t[:]
        state_t_2 = state[6*self.num_service_type:12*self.num_service_type]
        # 后面10 到 15*self.num_service_type 作为最旧时隙的状态舍去

        next_state = np.concatenate((state_t_2, state_t_1, state_t))
        self.state_t = next_state

        # print("reward",reward)#?usage_R:空间利用率，loss2：上带宽利用率，loss3：下带宽利用率，loss4：cpu利用率
        return next_state, reward, done, (delay/self.num_task), (energy/self.num_task), usage_R, loss2, loss3, loss4

    def sample(self):
        dict_a = self.action_space.sample()
        caching = dict_a["caching"]
        offloading = dict_a["offloading"]
        # uplink=dict_a["uplink"]
        # downlink=dict_a["downlink"]
        # frequency=dict_a["frequency"]
        action = np.concatenate((caching, offloading))
        return action

    def reset(self):
        self.current_steps = 0
        print("start reset!!!!!!!")

        user_distances = self.user_distance_sampler.get_samples(
            n_samples=self.num_task, convert_type='float')  # 抽取40个用户的距离
        print("user_distances", user_distances)
        self.channel_matrix = self.channel_generator.generate_channel_matrix(
            user_distances)  # 生成信道矩阵,40个用户，200个时隙

        state = np.zeros(
            (self.num_service_type*6*self.history_length,))  # len=150
        # ?因为是一条state含有三条历史，所以走完三步才有一条完整的state state=[st,st-1,st-2] len(st)=5*num_service_type=50
        # (input_size_service,output_size_service,computing_service_current,num_times_current)

        s_t = self.gen_new_state()

        # 生成上一时隙的caching决策当做当前时隙的状态
        caching_decision = self.action_space.sample()["caching"]

        #!将caching decision合存储约束化
        left_space_R_flag = self.ap_storage_capacity  # 用过了一次，复位
        # 计算当前caching decision的缓存空间利用率，并调整cahing decision
        for i in range(self.num_service_type):
            #!能缓存尽量缓存，无法缓存了置决策为0
            # 缓存服务，减去空间
            if caching_decision[i] == 1 and left_space_R_flag >= self.service_space_arry[i]:
                # 这保证了left_space_R>=0
                left_space_R_flag -= self.service_space_arry[i]
                caching_decision[i] = 1
            else:
                caching_decision[i] = 0

        s_t = np.concatenate((s_t, caching_decision))
        state[-self.num_service_type*6:] = s_t
        for i in range(self.history_length):  # 这三步是为了构造一个完整的state
            # (input_size_service,output_size_service,computing_service_current,num_times_current,caching)
            a_t = self.sample()
            # print(len(state))
            state, _, _, _, _, _, _, _, _ = self.step(a_t, state)
        print("end reset!!!!!!!")
        return state

    def seed(self, seed):
        self.action_space.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def gen_new_state(self):
        task_array = gen_40_task_zipf(
            size=self.num_task, service_sizes=self.num_service_type)
        # num_his = self.num_service_history[:]
        # compute_his = self.computing_service_history[:]

        # 记录当前state下的每个service type出现次数
        num_times_current = np.zeros((self.num_service_type,))
        # 当前state每类型任务的计算量
        input_size_service = np.zeros(
            (self.num_service_type,))  # 当前state每个任务的输入数据大小
        output_size_service = np.zeros(
            (self.num_service_type,))  # 当前state每个任务的输出数据大小
        computing_service_current = np.zeros(
            (self.num_service_type,))  # 当前state每个类型任务的总计算需求
        efficiencies_current = np.zeros(
            (self.num_service_type,))  # 当前state每个类型任务的频谱效率总计算需求
        print("shape channel_matrix", self.channel_matrix.shape)
        print("current_steps", self.current_steps)

        efficiencies = gen_efficiency(
            channel_gain=self.channel_matrix[:, self.current_steps], trans_power=self.trans_power)  # 40个用户的频谱效率

        for i, task in enumerate(task_array):
            # *类型是1到10,循环是1到11(不包括10)
            for service_type in range(1, self.num_service_type + 1):
                if task.type == service_type:
                    num_times_current[service_type - 1] += 1
                    computing_service_current[service_type -
                                              1] += task.require_resource
                    input_size_service[service_type - 1] += task.input_size
                    output_size_service[service_type - 1] += task.output_size
                    efficiencies_current[service_type - 1] += 1/efficiencies[i]

        s_t = np.concatenate(
            (input_size_service[:], output_size_service[:], computing_service_current[:],  num_times_current[:], efficiencies_current[:]))
        # print("s_t",s_t)
        return s_t


# env = Environment()
# state = env.reset()
# print(state,len(state))
# action_space = env.action_space
# # (caching,offloading,uplink,downlink,frequency)
# caching_offloading_low=np.zeros((20,))
# caching_offloading_high=np.ones((20,))
# low=np.concatenate((caching_offloading_low,action_space["uplink"].low,action_space["downlink"].low,action_space["frequency"].low))
# high=np.concatenate((caching_offloading_high,action_space["uplink"].high,action_space["downlink"].high,action_space["frequency"].high))
# action_range = [low, high]
# print(action_range)


def test_random_act(num_service_type=10):

    bandwidthup_space_low = np.ones((num_service_type,))
    bandwidthup_space_high = np.ones((num_service_type,)) * 4.0  # 3mhz

    bandwidthdown_space_low = np.ones((num_service_type,))
    bandwidthdown_space_high = np.ones((num_service_type,)) * 4.0  # 3mhz

    compute_resource_allocate_low = np.ones((num_service_type,))
    compute_resource_allocate_high = np.ones((num_service_type,)) * 6.0  # 3ghz

    action_space = spaces.Dict({'offloading': spaces.MultiBinary(10), 'caching': spaces.MultiBinary(10),
                                'uplink': spaces.Box(low=bandwidthup_space_low, high=bandwidthup_space_high), 'downlink': spaces.Box(low=bandwidthdown_space_low, high=bandwidthdown_space_high),
                                'frequency': spaces.Box(low=compute_resource_allocate_low, high=compute_resource_allocate_high)})

    at = action_space.sample()
    print(at)
