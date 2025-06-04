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
        ap_storage_capacity=300, 
        spectral_efficiency=3,  
        bandwith=20, 
        ap_computing_capacity=20,  
        cloud_computing_rate=5, 
        ap2c_rate=15,  
        history_length=3,
        seed=2023
    ):
        super(Environment, self).__init__()

        self.num_service_type = num_service_type  

        self._max_episode_steps = 201  # 一个episode 100步
        self.current_steps = 0
        # QoE-Aware Decentralized Task Offloading andResource Allocation for End-Edge-CloudSystems:A Game-Theoretical Approach
        self.trans_power = 1e-3*pow(10., 30./10.)
        self.num_task = num_task

        # *环境相关
        self.bandwith_up = bandwith  
        self.bandwidth_down = bandwith
        self.spectral_efficiency = spectral_efficiency  
        self.ap_storage_capacity = ap_storage_capacity  #
        self.ap_computing_capacity = ap_computing_capacity * \
            1000  
        self.cloud_computing_rate = cloud_computing_rate*1000  
        self.ap2c_rate = ap2c_rate  

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
        
        self.service_space_arry = gen_service_space(num_types=num_service_type)
        
        self.action_space = spaces.Dict(
            {'offloading': spaces.MultiBinary(self.num_service_type), 'caching': spaces.MultiBinary(self.num_service_type)})

        
        self.state_t = np.zeros((self.num_service_type*6*self.history_length,))
        
        self.cycle_proportion = np.ones(
            (self.num_service_type,))  
        
    def step(self, action, state):


        action = np.round(action)  
       
        caching_tplus1 = action[: self.num_service_type * 1][:]
        offloading = action[self.num_service_type *
                            1:self.num_service_type * 2][:]

       
        s_t = state[-self.num_service_type * 6:][:]  
        input_size_service = s_t[:self.num_service_type] 
        output_size_service = s_t[self.num_service_type *
                                  1:self.num_service_type*2]  
        computing_service_current = s_t[self.num_service_type *
                                        2:self.num_service_type*3] 
        efficiency = s_t[-self.num_service_type*2:-self.num_service_type:]
        caching_t = s_t[-self.num_service_type:]

     

        for i in range(len(offloading)): 
            if offloading[i] > caching_t[i]:
                offloading[i] = 0

       
        left_space_R_flag = self.ap_storage_capacity  
        for i in range(self.num_service_type):
            
            if caching_tplus1[i] == 1 and left_space_R_flag >= self.service_space_arry[i]:
                
                left_space_R_flag -= self.service_space_arry[i]
                
            else:
                caching_tplus1[i] = 0

     
        usage_R = 1-(left_space_R_flag/self.ap_storage_capacity)
        loss1 = usage_R  
      
        
       
        cost, loss2, loss3, loss4, delay, energy, _, _, _ = cvxpy_loss(caching_t, offloading, input_size_service, output_size_service, computing_service_current, variable_num=self.num_service_type,
                                                                       ig=efficiency, upper_bandwidth=self.bandwith_up, upper_cpu=self.ap_computing_capacity, e2c_rate=self.ap2c_rate,
                                                                       cloud_computing_rate=self.cloud_computing_rate, cycle_proportion=self.cycle_proportion, trans_power=self.trans_power)
        reward = (
            -(cost/self.num_task)
           
            + 0.5*(loss1+loss4)+0.1*(loss2+loss3)
        )

      

        self.current_steps += 1
        print("current_steps", self.current_steps)
        print("max_steps", self._max_episode_steps)
        done = True if self.current_steps == (
            self._max_episode_steps+self.history_length) else False
       
        state_t = np.concatenate((self.gen_new_state(), caching_tplus1[:]))
        state_t_1 = s_t[:]
        state_t_2 = state[6*self.num_service_type:12*self.num_service_type]
        
        next_state = np.concatenate((state_t_2, state_t_1, state_t))
        self.state_t = next_state

        
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
            n_samples=self.num_task, convert_type='float')  
        print("user_distances", user_distances)
        self.channel_matrix = self.channel_generator.generate_channel_matrix(
            user_distances)  

        state = np.zeros(
            (self.num_service_type*6*self.history_length,))  # len=150
       
        s_t = self.gen_new_state()

       
        caching_decision = self.action_space.sample()["caching"]

       
        left_space_R_flag = self.ap_storage_capacity  
        for i in range(self.num_service_type):
           
            if caching_decision[i] == 1 and left_space_R_flag >= self.service_space_arry[i]:
                
                left_space_R_flag -= self.service_space_arry[i]
                caching_decision[i] = 1
            else:
                caching_decision[i] = 0

        s_t = np.concatenate((s_t, caching_decision))
        state[-self.num_service_type*6:] = s_t
        for i in range(self.history_length):  
          
            a_t = self.sample()
          
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
        
        num_times_current = np.zeros((self.num_service_type,))
       
        input_size_service = np.zeros(
            (self.num_service_type,))  
        output_size_service = np.zeros(
            (self.num_service_type,)) 
        computing_service_current = np.zeros(
            (self.num_service_type,)) 
        efficiencies_current = np.zeros(
            (self.num_service_type,)) 
        print("shape channel_matrix", self.channel_matrix.shape)
        print("current_steps", self.current_steps)

        efficiencies = gen_efficiency(
            channel_gain=self.channel_matrix[:, self.current_steps], trans_power=self.trans_power) 

        for i, task in enumerate(task_array):
            
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



