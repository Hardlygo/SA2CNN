
import numpy as np
import cvxpy as cp
import mosek


def cvxpy_loss(caching_decision, offloading_decision, input_arry, output_arry, cpu_arry, ig, trans_power, variable_num=10, e2c_rate=15, cloud_computing_rate=5, upper_bandwidth=20, upper_cpu=20, cycle_proportion=330, BETA=0.1):
    '''
    param:
        caching_decision: 服务缓存决策
        offloading_decision: 服务卸载决策
        input_arry:输入数据量(Mb)
        output_arry:计算结果数据量(Mb)
        cpu_arry:cpu需求量
        ig:上行和下行的频谱效率
        e2c_rate: ES到Cloud的数据传输速度
        cloud_computing_rate: 云中心的CPU计算速度
        upper_bandwidth: 最多可分配带宽
        upper_cpu: ES最多可分配的CPU
        cycle_proportion:每个类型任务的CPU需求和输入的关系
        trans_power: 传输功率

    result:
        1. total delay
        2. usage uplink 上带宽利用率
        3. usage downlink 下带宽利用率
        4. ES的cpu利用率
        5. 每一步的上带宽分配数组
        6. 每一步的下带宽分配数组
        7. 每一步的frequency分配数组
    '''
    # for i in range(len(offloading_decision)): #不存在了只缓存，不卸载
    #     if offloading_decision[i]>caching_decision[i]:
    #         offloading_decision[i]=0

    k1 = caching_decision*offloading_decision
    k2 = 1-offloading_decision

    param1 = input_arry*ig
    param2 = output_arry*ig

    # cpu_arry/(1000*cycle_proportion)之前是除以1000现在为了减少数量悬殊直接除以10000
    param3 = cpu_arry*cycle_proportion/10000

    # print('param3', param3)
    print("caching_decision:", caching_decision)
    print("offloading_decision:", offloading_decision)
    print("input_arry:", input_arry)
    print("output_arry:", output_arry)
    print("cpu_arry:", cpu_arry)
    print("ig:", ig)
    print("e2c_rate:", e2c_rate)
    print("cloud_computing_rate:", cloud_computing_rate)
    print("upper_bandwidth:", upper_bandwidth)
    print("upper_cpu:", upper_cpu)
    print("cycle_proportion:", cycle_proportion)

    b = input_arry/e2c_rate+output_arry/e2c_rate+param3 / \
        (cloud_computing_rate/10000)  # 这个是已知的常量
    time_cloud = k2*b
    #!变量定义
    up_bandwidth_allocated = cp.Variable(variable_num)
    down_bandwidth_allocated = cp.Variable(variable_num)
    cpu_allocated = cp.Variable(variable_num)  # ?定义决策变量个数

    # 第一个问题，优化Cpu分配
    # a = cp.multiply(k1, cp.multiply(param3, cp.inv_pos(cpu_allocated)))
    a = cp.multiply(k1*param3,  cp.inv_pos(cpu_allocated))  # 先将已知的量相乘
    # ob = cp.sum(a)
    # print("curvature of ob1:", ob.curvature)

    # objective = cp.Minimize(ob)

    # constraints = [cpu_allocated >= 0, cp.sum(
    #     cpu_allocated) <= upper_cpu/10000]
    # print("constraints",constraints)
    # prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    # result = prob.solve(solver=cp.MOSEK)  # SCS
    # The optimal value for x is stored in x.value.
    # print("status:", prob.status) #status: optimal_inaccurate

    # print(cpu_allocated.value)
    # print("total cpu_allocated:", sum(cpu_allocated.value))
    # print(result)

    # 优化问题2 上下带宽资源分配
    bc = cp.multiply(k1, (cp.multiply(param1, cp.inv_pos(
        up_bandwidth_allocated))+cp.multiply(param2, cp.inv_pos(down_bandwidth_allocated))))
    c = cp.multiply(k2, (cp.multiply(param1, cp.inv_pos(up_bandwidth_allocated)) +
                    cp.multiply(param2, cp.inv_pos(down_bandwidth_allocated))))  # +b

    # 能耗
    d = (cp.multiply(param1, cp.inv_pos(
        up_bandwidth_allocated))+cp.multiply(param2, cp.inv_pos(down_bandwidth_allocated)))

    ob2 = cp.sum((1-BETA)*(c+bc+a)+BETA*trans_power * d)  # 优化目标为时延和能耗

    objective1 = cp.Minimize(ob2)

    constraints1 = [cpu_allocated >= 0, cp.sum(
        cpu_allocated) <= upper_cpu/10000, up_bandwidth_allocated >= 0, down_bandwidth_allocated >= 0, cp.sum(
        up_bandwidth_allocated) <= upper_bandwidth, cp.sum(down_bandwidth_allocated) <= upper_bandwidth]

    # print("constraints1",constraints1)
    prob1 = cp.Problem(objective1, constraints1)

    # The optimal objective is returned by prob.solve().
    result1 = prob1.solve(solver=cp.MOSEK)  # SCS ,verbose=True
    # The optimal value for x is stored in x.value.
    # print("status:", prob1.status)  # status: optimal_inaccurate
    # print(up_bandwidth_allocated.value)
    # print("total up_bandwidth_allocated:", sum(up_bandwidth_allocated.value))
    # print(down_bandwidth_allocated.value)
    # print("total down_bandwidth_allocated:",
    #       sum(down_bandwidth_allocated.value))
    # 计算总时延
    delay_edge = np.sum(k1 * param3 / (cpu_allocated.value + 1e-6))  # 边缘计算时延
    delay_transmission = np.sum((param1 / (up_bandwidth_allocated.value + 1e-6) +
                                param2 / (down_bandwidth_allocated.value + 1e-6)))  # 传输时延
    delay_cloud = np.sum(time_cloud)  # 云计算时延
    total_delay = delay_edge + delay_transmission + delay_cloud

    # 计算总能耗
    energy_transmission = np.sum(trans_power * (param1 / (up_bandwidth_allocated.value +
                                 1e-6) + param2 / (down_bandwidth_allocated.value + 1e-6)))  # 传输能耗
    total_energy = energy_transmission

    # 打印结果
    print("Edge delay:", delay_edge)
    print("Wireless transmission delay:", delay_transmission)
    print("Cloud delay:", delay_cloud)
    print("Total delay:", total_delay)
    print("Transmission energy:", energy_transmission)
    print("Total energy:", total_energy)
    # print(result1)
    # print("computing time:", result)
    cost = result1+(1-BETA)*np.sum(time_cloud)
    print("cost:", cost)
    return cost, sum(up_bandwidth_allocated.value)/upper_bandwidth, sum(down_bandwidth_allocated.value)/upper_bandwidth, sum(cpu_allocated.value*k1)/(upper_cpu/10000), total_delay, total_energy, up_bandwidth_allocated.value, down_bandwidth_allocated.value, cpu_allocated.value


# caching=np.array([1, 1 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0])
# offloading=np.array([1, 1 ,0 ,0 ,0 ,0 ,0, 0 ,0 ,0])

# Z_U=np.array([374. ,   115.   ,  78.   ,   0.  ,    0.  ,    0.   ,   0.  ,    0.   ,   0.,90.])
# Z_D=np.array([ 390.   ,  63.   ,  55. ,    17. ,    23.  ,    0. ,    18. ,    14., 0.  ,   20.])
# W=np.array([51.75 ,  15.375  , 5.25 ,   0.    ,  3.625  , 2.375  , 3.625, 0.   ,   0.    ,  0. ])
# ig=3
# dalay,loss1,loss2,loos3=cvxpy_loss(caching,offloading,Z_U,Z_D,W)

# #print(dalay,loss1,loss2,loos3)
