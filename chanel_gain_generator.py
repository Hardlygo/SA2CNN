
import math
import numpy as np
import scipy.special
from typing import Tuple, Optional, Union


class DynamicDistanceChannelGenerator:
    """支持动态用户距离输入的信道生成器"""

    def __init__(self,
                 num_users: int = 5,
                 time_slots: int = 100,
                 doppler_freq: float = 10.0,
                 shadowing_std: float = 8.0,
                 ts_duration: float = 20e-3,
                 seed: Optional[int] = None):
        """
        参数:
            num_users: 用户数量 (默认5)
            time_slots: 时隙数量 (默认100)
            doppler_freq: 多普勒频率(Hz) (默认10)
            shadowing_std: 阴影衰落标准差(dB) (默认8)
            ts_duration: 时隙持续时间(秒) (默认20ms)
            path_loss_exponent: 路径损耗指数 (默认3.7)
            seed: 随机种子 (默认None)
        """
        self.N = num_users
        self.T = time_slots
        self.fd = np.float32(doppler_freq)
        self.sigma_shadow = np.float32(shadowing_std)
        self.Ts = np.float32(ts_duration)

        if seed is not None:
            np.random.seed(seed)

    def calculate_path_loss1(self, distances: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算路径损耗（支持动态距离输入）
        参数:
            distances: 用户距离数组（长度需等于num_users）
        返回:
            线性域的路径损耗系数
        """
        distances = np.asarray(distances, dtype=np.float32)
        if len(distances) != self.N:
            raise ValueError(f"距离数组长度必须等于用户数{self.N}")

        """
        3GPP Urban Macro路径损耗模型 + 对数正态阴影
        PL = 120.9 + 37.6*log10(d[km]) + Xσ, Xσ~N(0,σ²)
        返回线性路径损耗 (不是dB值)
        """
        # 转换为千米
        d_km = distances  # / 1000.0

        # 对数正态阴影 (转换为线性域)
        shadowing = np.random.lognormal(
            mean=0,
            sigma=self.sigma_shadow,
            size=(self.N,))

        # 3GPP模型计算 (转换为线性值)
        pl_dB = 120.9 + 37.6 * np.log10(d_km)
        pl_linear = shadowing * np.power(10, -pl_dB / 10.0)

        return pl_linear

    def calculate_path_loss(self, distances: Union[np.ndarray, list]) -> np.ndarray:
        """
        计算路径损耗（支持动态距离输入）
        参数:
            distances: 用户距离数组（长度需等于num_users）
        返回:
            线性域的路径损耗系数
        """
        distances = np.asarray(distances, dtype=np.float32)
        if len(distances) != self.N:
            raise ValueError(f"距离数组长度必须等于用户数{self.N}")

        """
        3GPP Urban Macro路径损耗模型 + 对数正态阴影
        PL = -30-22log10(d)-22log10(fc)
        返回线性路径损耗 (不是dB值)
        """
        # 转换为千米
        d_km = distances * 1000

        pl_dB = 30 + 22 * np.log10(d_km)+20*math.log(4.9)
        pl_linear = np.power(10, -pl_dB/10)

        return pl_linear

    def _calculate_correlation(self) -> float:
        """计算时域相关系数（基于Jakes模型）"""
        x = 2 * np.pi * self.fd * self.Ts
        return np.float32(scipy.special.k0(x))

    def generate_time_correlated_fading(self) -> np.ndarray:
        """
        生成时域相关的瑞利衰落信道
        返回: (N用户, T时隙)的复数信道矩阵
        """
        H = np.zeros((self.N, self.T), dtype=np.complex64)
        rho = self._calculate_correlation()
        # noise_std = np.sqrt(1 - rho**2) * np.sqrt(0.5)

        # 初始状态
        H[:, 0] = np.sqrt(0.5*(np.random.randn(self.N)**2 +
                               np.random.randn(self.N)**2))
        # np.sqrt(0.5*(np.random.randn(self.N).astype(np.float32)**2 +
        #                          np.random.randn(self.N).astype(np.float32)**2))
        # np.sqrt(0.5) * (
        #     np.random.randn(self.N).astype(np.float32) +
        #     1j * np.random.randn(self.N).astype(np.float32))

        # 向量化时域演进
        for t in range(1, self.T):
            noise = np.sqrt(
                (1.-rho**2)*0.5*(np.random.randn(self.N)**2+np.random.randn(self.N)**2))

            H[:, t] = H[:, t-1] * rho + noise

        return H

    def generate_channel_matrix(self, distances: Union[np.ndarray, list]) -> np.ndarray:
        """
        生成完整的信道矩阵（需输入用户距离）
        参数:
            distances: 用户距离数组（长度=num_users）
        返回: (N用户, T时隙)的实数值信道增益矩阵
        """
        pl = self.calculate_path_loss(distances)
        h_small = self.generate_time_correlated_fading()
        # (np.abs(h_small)**2)
        return np.square(np.abs(h_small)) * pl[:, np.newaxis]


# 使用示例
if __name__ == "__main__":
    num_users = 40
    time_slots = 200
    # 创建信道生成器
    channel_gen = DynamicDistanceChannelGenerator(
        num_users=num_users,
        time_slots=time_slots,
        seed=2023
    )  # 决定了每个episode需要多少个距离记录：user_n*time_slot

    # 自定义用户距离（可动态改变）
    # user_distances = [150, 300, 450]  # 单位：米
    user_distances = np.random.uniform(
        low=50, high=150, size=(num_users, ))

    # 生成信道矩阵
    channel_matrix = channel_gen.generate_channel_matrix(user_distances)

    # 获取信道信息
    info = channel_gen.get_correlation_info()

    print("=== 信道参数 ===")
    print(f"多普勒频率: {info['doppler_freq']}Hz")
    print(f"时隙长度: {info['ts_duration']*1000}ms")
    print(f"阴影衰落标准差: {info['shadowing_std']:.4f}")
    print(f"时域相关系数: {info['correlation_coeff']:.4f}")

    print("\n=== 信道矩阵 ===")
    print("形状:", channel_matrix.shape)
    print("所有用户的前5时隙增益:")
    print(channel_matrix[:, :5])
