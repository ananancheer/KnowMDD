import csv
import itertools
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from config import Config
from data_loader import load_data
from load import Multi, Multi_sum
from model import GCN,KnowMDD
from seed import set_seed
from utils import FC_img, get_fname
from nilearn import datasets, plotting


import warnings
warnings.filterwarnings('ignore')


def train(args,model_name,data1,data2):
    print("========================strat train====================")
    # set_seed(args.seed)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=64)

    for fold, (train_idx, test_idx) in enumerate(kf.split(data1)):
        print(f"Fold {fold + 1}:")
        args.fold_num = fold+1
        print("fold :{} device {}".format(args.fold_num, args.device))
        
        train_data1 = [data1[i] for i in train_idx];train_data2 = [data2[i] for i in train_idx]
        test_data1 = [data1[i] for i in test_idx];test_data2 = [data2[i] for i in test_idx]

        print(f"shape of train: {len(train_data1)}")
        print(f"shape of test: {len(test_data1)}")

        train_loader_1 = DataLoader(train_data1, batch_size=args.batch_size, shuffle=True)
        train_loader_2 = DataLoader(train_data2, batch_size=args.batch_size, shuffle=True)
        test_loader_1 = DataLoader(test_data1, batch_size=args.batch_size, shuffle=False)
        test_loader_2 = DataLoader(test_data2, batch_size=args.batch_size, shuffle=False)
        # 定义model
        model = KnowMDD(args = args, in_feats = 64,hidden_size = 128,out_size = 32).to(args.device)
        # print(model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        train_loss = [];train_acc=[]
        history = []
        for epoch in range(args.num_epoch):
            model.train()
            epoch_loss = [];epoch_attweight1 = [];epoch_attweight2 = []
            for g1,g2 in zip(train_loader_1,train_loader_2):
                optimizer.zero_grad()
                cl_loss,out,att_weight1,att_weight2 = model(g1,g2)
                loss = cl_loss/4 + criterion(out, g1.y)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()) 
                epoch_attweight1.append(att_weight1)
                epoch_attweight2.append(att_weight2)
            all_preds = []; all_labels = []; 
            model.eval()
            with torch.no_grad():
                for g1,g2 in zip(test_loader_1,test_loader_2):
                    _,out,_,_ = model(g1,g2)
                    preds = torch.softmax(out, dim=1)
                    
                    all_preds.extend(np.argmax(preds.cpu().detach().tolist(), axis=1))
                    all_labels.extend(g1.y.cpu().detach().tolist())
            
            epoch_losses =  sum(epoch_loss)/len(train_loader_1)
            avg_weight1 = torch.mean(torch.stack(epoch_attweight1), dim=0)
            avg_weight2 = torch.mean(torch.stack(epoch_attweight2), dim=0)
            epoch_acc = accuracy_score(all_labels, all_preds)*100
            sensitivity = recall_score(all_labels, all_preds)*100
            tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
            specificity = (tn / (tn + fp))*100
            epoch_f1 = f1_score(all_labels, all_preds, average='macro')*100
            if (epoch+1)%args.epoch_check == 0:
                train_loss.append(epoch_losses)
                train_acc.append(epoch_acc)
            history.append({'epoch': epoch + 1, 'loss': epoch_losses, 'accuracy': epoch_acc,'sen':sensitivity,'spec':specificity, 'f1': epoch_f1})
            print(f'Epoch [{epoch + 1}], Loss: {epoch_losses:.4f}, Accuracy: {epoch_acc:.4f},SEN: {sensitivity},SPEC: {specificity}, F1: {epoch_f1:.4f}')

            save_dire = f"checkpoints/model/{model_name}/{args.multi_atlas[0]}_{args.multi_atlas[1]}/"
            os.makedirs(f"{save_dire}", exist_ok=True)
            file_exists = os.path.isfile(f"{save_dire}training_history_fold{args.fold_num}.csv")
            with open(f'{save_dire}training_history_fold{args.fold_num}.csv', 'a', newline='') as csvfile:
                fieldnames = [f"{args.multi_atlas[0]}",f'{args.multi_atlas[1]}']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({f"{args.multi_atlas[0]}": avg_weight1.cpu().detach().numpy(),
                                 f"{args.multi_atlas[1]}": avg_weight2.cpu().detach().numpy()})
                
        save_dire = f"result/model/{model_name}/{args.multi_atlas[0]}_{args.multi_atlas[1]}/"
        os.makedirs(f"{save_dire}", exist_ok=True)
        with open(f'{save_dire}training_history_fold{args.fold_num}.csv', 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'loss', 'accuracy', 'sen', 'spec', 'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        print(f"result saved in '{save_dire}'")
        
        # 绘制每个fold的损失曲线
        os.makedirs(f"{save_dire}loss/", exist_ok=True)
        plt.figure()
        plt.plot(range(len(train_loss)), np.array(train_loss), linestyle='-',color='b',label = 'Loss')
        plt.title(f'Fold {fold + 1} Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.grid(True)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(f"{save_dire}loss/", f"fold_{fold+1}_loss_curve.png"))
        plt.close()
        print(f"loss graph saved in '{save_dire}loss/'")
        # 绘制每个fold的准确性曲线
        os.makedirs(f"{save_dire}acc/", exist_ok=True)
        plt.figure()
        plt.plot(range(len(train_acc)), np.array(train_acc), linestyle='-',color='y',label = 'acc')
        plt.title(f'Fold {fold + 1} Training ACC Curve')
        plt.xlabel('Epoch')
        plt.ylabel('ACC')

        plt.grid(True)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(f"{save_dire}acc/", f"fold_{fold+1}_acc_curve.png"))
        plt.close()
        print(f"loss graph saved in '{save_dire}acc/'")
    print("========================train end====================")



if __name__ == '__main__':
    
    args = Config()
    atlas_list = ['AAL',"Harvard",'Craddock']
    num_ROI_list = [116,112,200]
    # atlas_list = ["Harvard",'Craddock']
    # num_ROI_list = [112,200]
    node_pairings = list(itertools.combinations(range(len(atlas_list)), 2))
    atlas_pairings = list(itertools.combinations(atlas_list, 2))
    # node_pairings = node_pairings[:-1];atlas_pairings = atlas_pairings[:-1]
    sum_list= []
    model = "KnowMDD_graphtest"
    text = 'KnowMDD_graph=KnowMDD_graphtest'
    file_names = get_fname()
    set_seed(args.seed)
    for node_pair in node_pairings:
        start_time = time.time()
        args.multi_atlas = [atlas_list[i] for i in node_pair]  
        args.multi_numROI = [num_ROI_list[i] for i in node_pair] 
        args.multi_keep = [args.keep_nodes[i] for i in node_pair]
        print(f"Atlas: {args.multi_atlas}")
        data_1 = load_data(args=args,atlas=args.multi_atlas[0],file_names = file_names,name=model)
        data_2 = load_data(args=args,atlas=args.multi_atlas[1],file_names = file_names,name=model)

        train(args,model,data_1,data_2)
        sum_result = Multi(model, args.multi_atlas,text)
        sum_list.append(sum_result)  # 将每次调用的结果合并
        print(f"total time: {(time.time()-start_time)/60} min")
    Multi_sum([model], sum_list, atlas_pairings)
