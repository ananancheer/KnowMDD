import csv
import itertools
import os
import numpy as np
import pandas as pd

def compute_sum(save_row, atlas):
    # 将列表转为 DataFrame 以便于计算
    save_row_df = pd.DataFrame(save_row)

    # 计算均值和标准差
    mean_acc = np.mean(save_row_df[0])
    std_dev_acc = np.std(save_row_df[0])
    formatted_acc = f"{mean_acc:.2f}±{std_dev_acc:.3f}"

    mean_sen = np.mean(save_row_df[1])
    std_dev_sen = np.std(save_row_df[1])
    formatted_sen = f"{mean_sen:.2f}±{std_dev_sen:.3f}"

    mean_spec = np.mean(save_row_df[2])
    std_dev_spec = np.std(save_row_df[2])
    formatted_spec = f"{mean_spec:.2f}±{std_dev_spec:.3f}"

    mean_f1 = np.mean(save_row_df[3])
    std_dev_f1 = np.std(save_row_df[3])
    formatted_f1 = f"{mean_f1:.2f}±{std_dev_f1:.3f}"

    # 创建汇总行
    sum_row = [formatted_acc,formatted_sen,formatted_spec,formatted_f1]
    return sum_row


def Multi(model,atlas_list,text):
    sum_list = []
    atlas = f"{atlas_list[0]}_{atlas_list[1]}"
    save_dir = f"result/model/{model}/{atlas}/training_history_fold"
    files = [f'{save_dir}1.csv', f'{save_dir}2.csv', f'{save_dir}3.csv', f'{save_dir}4.csv', f'{save_dir}5.csv']
    data,max_row,file_row = max_sum(files)

    # 确保目录存在
    save_dir = f"result/model/{model}/{atlas}"
    file_exists = os.path.isfile(f"{save_dir}/load.csv")
    # 保存到 load.csv
    with open(f'{save_dir}/load.csv', 'w', newline='',encoding='UTF-8') as csvfile:
        fieldnames = ["fold","accuracy",'sen', 'spec', 'f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(files)):
            writer.writerow({
            "fold": i,
            "accuracy": data[i,0],
            "sen": data[i,1],
            "spec": data[i,2],
            "f1": data[i,3]
            })
        file_row = compute_sum(data,atlas)
        writer.writerow({
            "fold": 'sum',
            "accuracy": file_row[0],
            "sen": file_row[1],
            "spec": file_row[2],
            "f1": file_row[3]
            })
        csvfile.write("\n")
        # csvfile.write("mean",formatted_loss,std_dev_acc,formatted_f1)
                
    print(f"Succeed load in {save_dir},最大epoch为max_row")
    save_dir = f"result/sum/"
    os.makedirs(f"{save_dir}", exist_ok=True)
    file_exists = os.path.isfile(f"{save_dir}{atlas}.csv")
    with open(f'{save_dir}{atlas}.csv', 'a', newline='',encoding='UTF-8') as file:
        fieldnames = ["model","accuracy",'sen', 'spec', 'f1','text']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "model": model,
            "accuracy": file_row[0],
            "sen": file_row[1],
            "spec": file_row[2],
            "f1": file_row[3],
            'text':text
            })
    print(f"Succeed load in {save_dir}/{atlas}.csv")
    return file_row

def Multi_sum(model_list, sum_result, atlas_pairings):
    for model in model_list:
        save_dir = f"result/model/{model}"
        with open(f'{save_dir}/sum.csv', 'w', newline='', encoding='UTF-8') as csvfile:
            fieldnames = ["atlas", "accuracy", "sen", "spec", "f1"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, row in enumerate(atlas_pairings):
                atlas = f"{atlas_pairings[i][0]}_{atlas_pairings[i][1]}"
                writer.writerow({
                    'atlas': atlas,
                    'accuracy': sum_result[i][0],
                    'sen': sum_result[i][1],
                    'spec': sum_result[i][2],
                    'f1': sum_result[i][3]
                })
    print(f"Succeed load in {save_dir}/sum.csv")

def max_sum(files):
    data = []
    for file_path in files:
        file_data = pd.read_csv(file_path)
        data.append(file_data[['accuracy','sen','spec','f1']])
    data = np.array(data)
    row_sums = np.sum(data, axis=0)/ len(files)

    accuracy = row_sums[:, 0]
    f1 = row_sums[:, -1]

    max_indices = np.where(accuracy == np.max(accuracy))[0]

    if len(max_indices) > 1:
        max_row_index = max_indices[np.argmax(f1[max_indices])]
    else:
        max_row_index = max_indices[0]

    save_data = data[:,max_row_index,:]


    return save_data,max_row_index,row_sums[max_row_index]



if __name__ == '__main__':
    

    # atlas_list = ['AAL',"Harvard",'Craddock']
    # num_ROI_list = [116,112,200]
    atlas_list = ["Harvard",'Craddock']
    num_ROI_list = [112,200]
    node_pairings = list(itertools.combinations(range(len(atlas_list)), 2))
    atlas_pairings = list(itertools.combinations(atlas_list, 2))
    # node_pairings = node_pairings[:-1];atlas_pairings = atlas_pairings[:-1]
    sum_list= []
    model = "KonwMDD_mlp"
    text = 'KonwMDD_mlp'
    for node_pair in node_pairings:
        multi_atlas = [atlas_list[i] for i in node_pair]  
        print(f"Atlas: {multi_atlas}")
        sum_result = Multi(model, multi_atlas ,text)
        sum_list.append(sum_result)  # 将每次调用的结果合并