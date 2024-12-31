import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def flatten_fc(data):
    x, y = np.triu_indices(data.shape[1], k=1)  #x 和 y，分别表示上三角矩阵的行和列索引
    FC_flatten = data[:, x, y]
    return FC_flatten
    
def flatten2dense(flatten, ROI):
    # 创建一个ROI_num * ROI_num 的矩阵
    x, y = np.triu_indices(ROI, k=1)
    sym = np.zeros((ROI, ROI))
    sym[x, y] = flatten
    sym = sym + sym.T
    return sym

def get_fname():
    data_load_path = f"data/ROISignals_FunImgARCWF"
    #读取全部list
    f_names = os.listdir(data_load_path)
    f_names = [file for file in f_names if 'S20' in file]
    f_names = [file for file in f_names if file not in ['ROISignals_S20-2-0095.mat', 'ROISignals_S20-1-0251.mat']]

    return f_names

def get_fname_f21():
    data_load_path = f"data/ROISignals_FunImgARCWF"
    #读取全部list
    f_names = os.listdir(data_load_path)
    f_names = [file for file in f_names if 'S21' in file]

    return f_names

def get_fname_f25():
    data_load_path = f"data/ROISignals_FunImgARCWF"
    #读取全部list
    f_names = os.listdir(data_load_path)
    f_names = [file for file in f_names if 'S25' in file]

    return f_names
