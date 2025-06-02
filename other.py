import numpy as np
import random
import math




def gen_input_ouput_data(tasks=40):
    """
    根据泊松分布生成每个任务的输入输出任务数据大小
    到达率 λ∈[8, 12, 18, 24]Mb
    """
    sizes = np.random.uniform(low=2.4, high=11.2, size=(
        tasks, 2))  
    sizes = sizes.reshape(tasks * 2)  
    np.random.shuffle(sizes)
    #     print(sizes)
    sizes = sizes.reshape((tasks, 2))

    return sizes





def gen_service_space():
    """
    10个服务类型占用空间服从均匀分布
    只需要生成一次？因为占用空间在一个episode内不改变的
    40 80 G
    """
   
    spaces = np.random.uniform(40, 81, size=(10,))  #
    
    spaces = np.around(spaces)  
    print(spaces, spaces.sum())
    return spaces





class Task(object):
    def __init__(self, input_size, output_size, service_type, require_resource=None):
        """
        任务包含 （输入数据，结果数据，计算需求，所属类型）
        """
        super().__init__()
        self.input_size = input_size  # * 1e6  # Mb化成b
        self.output_size = output_size  # * 1e6
        self.type = service_type
        if require_resource is None:
            self.require_resource = (self.input_size)
        else:
            self.require_resource = require_resource

    def __str__(self):
        return "(input:{},output:{},require_computing:{},type:{})".format(
            self.input_size, self.output_size, self.require_resource, self.type
        )


SERVICE_TYPE = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]






def Zipf_sample(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    #  离散采样来模拟截断的Zipf
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max+1)  # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)            # normalized

    return np.random.choice(v, size=size, replace=True, p=p)


def gen_efficiency(channel_gain, trans_power):

    noise_power = 1e-3*pow(10., -174./10.)  # 174 #2*1e-13#-114

    efficiency2server_M = np.log2(
        1+((trans_power*channel_gain) / (noise_power)))  # +interfrence
    return efficiency2server_M




def gen_40_task_zipf(size=40, service_sizes=12):
    """
    生成目标数量的task数组
    """
    data_size_arry = gen_input_ouput_data(size)  # 生成40个任务
    computation_density_array = np.random.uniform(500, 900, size=(
        size, )) 
    #!两种不一样的选服务类型的方法
    #!泽夫分布
    service_types = Zipf_sample(
        1.2, min=1, max=service_sizes, size=size)  # 与服务类型个数一致
    #!指定概率
    arr_len = len(data_size_arry)
    task_arry = [] 
    for idx in range(arr_len):
        if len(task_arry) < arr_len:
            task_arry.append(None)
        task = Task(
            input_size=data_size_arry[idx][0], output_size=data_size_arry[idx][
                1], service_type=service_types[idx], require_resource=computation_density_array[idx]  # * 330
        )
        task_arry[idx] = task

    return task_arry












