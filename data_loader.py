from datetime import datetime
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import io, stats
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops,to_scipy_sparse_matrix,to_networkx
from config import Config
from utils import flatten2dense, flatten_fc, get_fname
import torch.nn.functional as F


def get_subject_id(file_name):
    mat_full_name = str(file_name)  # file_name : ROISignals_S1-1-0001.mat
    file_name_label = file_name.split('-')[1]

    if os.path.splitext(mat_full_name)[1] == ".npy":
        mat_full_name = mat_full_name.replace('.npy', '.mat')

    if file_name_label == "1":
        label = '1'
        subject_ID = str(file_name[11:-4])
    elif file_name_label == "2":
        label = '0'
        subject_ID = str(file_name[11:-4])  # S1-1-0001
    else:
        print('where is symptoms!')
        sys.exit()
    return label, subject_ID, mat_full_name  # MDD, S1-1-0001, ROISignals_S1-1-0001.mat

def get_FC_map(file_names,data_load_path):
    data = []
    label = []
    data_score = []
    train_weight_index = [0, 0]
    f_name = []
    sheets = pd.read_excel('data/REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx', sheet_name=["MDD", "Controls"])
    for file_name in file_names:
        symptom, subject_ID, mat_full_name = get_subject_id(file_name)
        mat = io.loadmat(data_load_path + "/" + mat_full_name)
        mat = mat['ROI_Functional_connectivity']
        mat[np.isinf(mat)] = 1
        # mat[np.isnan(mat)] = 0
        if symptom == "1":
            csv = sheets["MDD"]
            # score = Hamd.loc[Hamd["ID"] == subject_ID, "HAMD"].values
            # if len(score) == 0 or score == '[]':
            #     continue
            # # label.append(score[0])
            label.append(1)
            train_weight_index[1] += 1
        else:
            csv = sheets["Controls"]
            label.append(0)
            train_weight_index[0] += 1
        
        score = csv.loc[csv["ID"] == subject_ID, ["Sex","Age","Education (years)"]].values[0]
        # print(subject_ID,score_append.shape)
        # data.append(mat)
        data.append(mat)
        data_score.append(score)
    data_score = np.array(data_score)
    [data, label] = [np.array(data), np.array(label)]
    # scaler = StandardScaler()
    # data_score = scaler.fit_transform(data_score)
    # data_score = F.normalize(data_score, p=2, dim=1)
    data_score_expanded = np.repeat(data_score[:, np.newaxis, :], data.shape[1], axis=1)
    data_concat = np.concatenate([data, data_score_expanded], axis=2)
    # score_append = np.hstack((mat, np.tile(data_score, (len(data), 1))))
    # print(f"Length of graph_weight: {train_weight_index}")

    return [data, label, train_weight_index,data_concat]

def make_topology(args,atlas,data,labels,name):
    t_test_save_dir = f'Topology/{atlas}'

    if os.path.exists(f"{t_test_save_dir}/{atlas}_{name}.npy"):
        p_value = np.load(f"{t_test_save_dir}/{atlas}_{name}.npy")
        return p_value
    
    ROI = data.shape[-1]
    flatten = flatten_fc(data)
    MDDflatten = flatten[labels == 1]
    NCflatten = flatten[labels == 0]
    
    p_value = stats.ttest_ind(MDDflatten, NCflatten, equal_var=False).pvalue
    p_value = flatten2dense(p_value, ROI)
    if not (os.path.isdir(t_test_save_dir)):
        os.makedirs(t_test_save_dir)
    
    np.save(f"{t_test_save_dir}/{atlas}_{name}.npy", p_value)

    return p_value

def define_node_edge(data, topology,p_value,edge_binary,edge_abs):
    feature = data.copy()
    mask = topology <= p_value
    if edge_abs:
        feature = np.abs(feature)
    feature = feature * mask
    if edge_binary:
        feature[feature != 0] = 1
    static_edge = feature
    return static_edge

def create_pyg_data(node_tensor, edge_tensor, label_tensor):
    edge_index = torch.nonzero(edge_tensor, as_tuple=False).t().contiguous()
    x = node_tensor.clone().detach()  
    y = label_tensor.clone().detach()  
    data = Data(x=x, edge_index=edge_index, y=y.long())
    edge_index, edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
    return Data(x=x, edge_index=edge_index, y=y.long())

def load_data(args,atlas,file_names,name):
    graph = []
    data_load_path = f"data/{atlas}/{atlas}_FC"
    #读取全部list
    [data, labels, _,data_concat] = get_FC_map(file_names,data_load_path)
    topology = make_topology(args,atlas,data,labels,name)
    static_edge = define_node_edge(data=data, topology=topology, p_value=0.05, edge_binary=True, edge_abs=True)
    
    Node_list = torch.FloatTensor(data_concat).to(args.device)
    A_list = torch.FloatTensor(static_edge).to(args.device)
    label = torch.LongTensor(labels).to(args.device)

    graph = [create_pyg_data(node, edge, label) for node, edge, label in zip(Node_list, A_list, label)]
    # print(f"Length of graph: {len(graph)}")        

    return graph


if __name__ == "__main__":
    arg = Config()

    atlas_List = ["Harvard"]
    numROI = [112]

    for i,atlas in enumerate(atlas_List):
        arg.atlas = atlas
        arg.numROI = numROI[i]
        f_names = get_fname()
        #读取全部list
        print(f"{arg.atlas}'s graph is loading,ROI number: {arg.numROI}")
        load_data(args=arg,atlas=atlas,file_names=f_names,name = "all")