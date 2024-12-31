import numpy as np
import pandas as pd
from scipy import io
import os, sys

import time

from utils import get_fname, get_fname_f25

# 计算FC矩阵
def make_fc_map(f_names, atlas="AAL"):
    
    data_save_dir = 'data/ROISignals_FCmap/{}_Kendall_FC'.format(atlas)
    if not (os.path.isdir(data_save_dir)):
        os.makedirs(data_save_dir)
    cont = 1
    for f_name in f_names:
        mat_file = io.loadmat('data/ROISignals_FunImgARCWF/{}'.format(f_name))
        mat = mat_file['ROISignals']  # 200*1833

        if atlas == "AAL":
            mat = mat[:, :116]   # AAL 116 ROI
        elif atlas == "Harvard":
            mat = mat[:, 116:228]  # Harvard Oxford 112 ROI
        elif atlas == "Craddock":
            mat = mat[:, 228:428]   # Craddock 200 ROI


        mat_transpose = np.array(mat).transpose()   #从每个ROI中提取BOLD信号的平均时间序列

        roi_dict = {}
        
        for i in range(len(mat_transpose)):
            roi = mat_transpose[i]
            roi_dict["{}".format(i)] = roi

        roi_df = pd.DataFrame(roi_dict)

        # correlation
        # corr_df = roi_df.corr(method='pearson')
        # corr_df_spearman = roi_df.corr(method='spearman')
        corr_df_kendall = roi_df.corr(method='kendall')

        corr_matrix = np.array(corr_df_kendall, dtype=[('ROI_Functional_connectivity', 'float64')])

        corr_matrix_dict = {}
        for varname in corr_matrix.dtype.names:
            corr_matrix_dict[varname] = corr_matrix[varname]

        # functional connectivity matrix
        io.savemat('data/ROISignals_FCmap/{}_Kendall_FC/{}'.format(atlas, f_name), corr_matrix_dict)
        print(cont); cont = cont+1

    return f"{data_save_dir}"

#  进行 Fisher Z 转换
#  为了使相关系数的分布更接近正态分布
def fisher_z_transformation(FC_data="", atlas="AAL"):

    data_save_dir = f'data/{atlas}/{atlas}_Kendall_FC'
    if not (os.path.isdir(data_save_dir)):
        os.makedirs(data_save_dir)
    path_dir = FC_data

    f_names = os.listdir(path_dir)
    print(len(f_names))


    for f_name in f_names:
        corr_matrix = io.loadmat('{}/{}'.format(FC_data, f_name))

        corr_matrix = corr_matrix['ROI_Functional_connectivity']
        corr_matrix = np.array(corr_matrix)

        for row in range(len(corr_matrix)):
            for col in range(len(corr_matrix)):
                rho = corr_matrix[row][col]
                corr_matrix[row][col] = (1.0 / 2.0) * (np.log((1.0 + rho) / (1.0 - rho)))


        corr_matrix_dict = {}
        corr_matrix = np.array(corr_matrix, dtype=[('ROI_Functional_connectivity', 'float64')])

        for varname in corr_matrix.dtype.names:
            corr_matrix_dict[varname] = corr_matrix[varname]

        io.savemat(f"{data_save_dir}/{f_name}", corr_matrix_dict)
    return f"{data_save_dir}"

if __name__ == "__main__":
    '''
    AAL 116 ROI
    Harvard Oxford 112 ROI
    Craddock 200 ROI
    '''
    f_names = get_fname()
    start_time = time.time()
    atlas_list = ['AAL',"Harvard",'Craddock']
    for atlas in atlas_list:
        print(f"==========={atlas} FC Map build=============")
        FC_data = make_fc_map(f_names, atlas=atlas)
        print("FC Map saved in ",FC_data)
        FC_data_fisher = fisher_z_transformation(FC_data=FC_data, atlas=atlas)
        print("fisher_z FC Map saved in ",FC_data_fisher)
        print(f"==========={atlas} FC Map build END=============")
    
    print("time: ", time.time() - start_time)