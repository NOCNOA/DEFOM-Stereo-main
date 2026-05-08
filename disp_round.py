import numpy as np
import torch
import re

def read_pfm(file_path):
    """读取 pfm 文件并返回 numpy 数组"""
    with open(file_path, 'rb') as f:
        # 1. 读取 Header
        header = f.readline().decode('utf-8').rstrip()
        if header == 'PF':
            channels = 3
        elif header == 'Pf':
            channels = 1
        else:
            raise ValueError("不是有效的 PFM 文件。")

        # 2. 读取 宽 和 高
        dims = f.readline().decode('utf-8').rstrip()
        width, height = map(int, dims.split())

        # 3. 读取 比例因子/端序
        scale = float(f.readline().decode('utf-8').rstrip())
        endian = '<' if scale < 0 else '>' # 小端序或大端序

        # 4. 读取 数据
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, channels) if channels == 3 else (height, width)
        
        # PFM 数据存储通常是倒置的 (从底向上)，所以需要翻转
        return np.flipud(data.reshape(shape)).copy()

def analyze_pfm_integer_ratio(file_path):
    # 加载数据
    arr = read_pfm(file_path)
    # 转换为 tensor 方便后续处理（也可以直接用 numpy）
    data = torch.from_numpy(arr)

    # 核心判断逻辑 (带 1e-6 容差)
    is_integer = (data - data.round()).abs() < 1e-1
    
    # 计算比例
    ratio = is_integer.float().mean().item()

    print(f"PFM 形状: {list(data.shape)}")
    print(f"整数占比: {ratio:.2%}")
    return ratio

# 使用示例
analyze_pfm_integer_ratio("/home/share/yijiayi/sceneflow/FlyingThings3D/disparity/TEST/B/0066/left/0012.pfm")