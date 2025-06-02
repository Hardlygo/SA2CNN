import os
import pandas as pd
import random
import numpy as np
from typing import List, Union, Optional

INPUT_DIR = 'data_source'
FILE = os.path.join(INPUT_DIR, '02merge_out.csv')


class DataSampler:
    def __init__(self,  target_column: str, csv_path: str = FILE, shuffle: bool = True):
        """
        改进版机器学习数据采样器
        :param csv_path: CSV文件路径
        :param target_column: 目标特征列名
        :param shuffle: 是否在初始化时打乱数据顺序
        """
        self.df = pd.read_csv(csv_path, sep=None, engine='python')
        self.target_column = target_column
        self._validate_column()  # 验证目标列是否存在

        # 原始数据指针
        self.pointer = 0
        self.values = self.df[self.target_column].tolist()

        if shuffle:
            self._shuffle_data()  # 默认 打乱数据顺序

    def _validate_column(self):
        """验证目标列是否存在"""
        if self.target_column not in self.df.columns:
            available_cols = [
                c for c in self.df.columns if not c.startswith('reserved')]
            raise ValueError(
                f"列 '{self.target_column}' 不存在。可用特征列: {available_cols}")

    def _shuffle_data(self):
        """随机打乱数据顺序"""
        random.shuffle(self.values)
        print(f"已随机打乱 {len(self.values)} 条'{self.target_column}'数据")

    def get_samples(self,
                    n_samples: int = 1,
                    convert_type: str = 'float',
                    reset_pointer: bool = False) -> np.ndarray:
        """
        获取指定数量的样本
        :param n_samples: 需要获取的样本数量
        :param convert_type: 返回数据类型(float/int/str)
        :param reset_pointer: 是否重置指针从数据开头开始
        :return: numpy数组
        """
        if reset_pointer:
            self.pointer = 0

        # 检查请求样本数是否合理
        if n_samples <= 0:
            raise ValueError("采样数量必须大于0")

        # 如果请求数量超过剩余数据量，循环使用数据
        if self.pointer + n_samples > len(self.values):
            available = len(self.values) - self.pointer
            samples = self.values[self.pointer:] + \
                self.values[:n_samples - available]
            self.pointer = n_samples - available  # 从头开始重新记录下标

            # samples =np.random.uniform(
            #     low=50, high=150, size=(n_samples, ))  # ?单位m
            # self.pointer = 0 #随机的做法
        else:
            samples = self.values[self.pointer:self.pointer + n_samples]
            self.pointer += n_samples

        return self._convert_samples(samples, convert_type)

    def get_random_samples(self,
                           n_samples: int = 1,
                           convert_type: str = 'float') -> np.ndarray:
        """
        随机抽取样本(不改变原始顺序)
        :param n_samples: 需要获取的样本数量
        :param convert_type: 返回数据类型(float/int/str)
        :return: numpy数组
        """
        if n_samples <= 0:
            raise ValueError("采样数量必须大于0")

        samples = random.sample(self.values, min(n_samples, len(self.values)))
        return self._convert_samples(samples, convert_type)

    def _convert_samples(self, samples: List, convert_type: str) -> np.ndarray:
        """内部使用的数据转换方法"""
        if convert_type == 'float':
            samples = [float(x) if str(x).replace(
                '.', '', 1).isdigit() else np.nan for x in samples]
            result = np.array(samples, dtype=np.float32)
            return result[~np.isnan(result)]
        elif convert_type == 'int':
            samples = [int(float(x)) if str(x).replace(
                '.', '', 1).isdigit() else 0 for x in samples]
            return np.array(samples, dtype=np.int32)
        return np.array(samples, dtype=np.str_)

    @property
    def total_samples(self) -> int:
        """返回总样本数"""
        return len(self.values)

    @property
    def remaining_samples(self) -> int:
        """返回剩余未采样数量"""
        return len(self.values) - self.pointer


# 使用示例
if __name__ == '__main__':
    # 初始化采样器
    sampler = DataSampler(
        target_column='distance_km',
        shuffle=True
    )

    # 示例1：顺序获取样本
    print("顺序采样示例:")
    for i in range(3):
        samples = sampler.get_samples(n_samples=5, convert_type='float')
        print(f"批次 {i+1}: {samples.shape}个样本 | 数据类型: {samples.dtype}")
        print(f"样本值: {samples}\n")

    # 示例2：随机获取样本
    print("\n随机采样示例:")
    random_samples = sampler.get_random_samples(
        n_samples=10, convert_type='float')
    print(f"随机获取 {random_samples.shape} 个样本")
    print(f"前5个值: {random_samples[:5]}...")

    # 示例3：重置指针后采样
    print("\n重置指针后采样:")
    reset_samples = sampler.get_samples(
        n_samples=8,
        convert_type='float',
        reset_pointer=True
    )
    print(f"重置后获取 {reset_samples.shape} 个样本")
    print(f"样本均值: {np.mean(reset_samples):.2f}")
