import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool





class GCN(torch.nn.Module):
    def __init__(self, args, in_feats, hidden_size,out_size):
        super(GCN,self).__init__()
        self.args = args
        self.in_feats =in_feats
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.multi_numROI = args.multi_numROI
        self.fcl = nn.Linear(self.multi_numROI[0]+3, in_feats)
        self.fcr = nn.Linear(self.multi_numROI[1]+3, in_feats)
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, out_size)
        self.fc1 = nn.Linear(hidden_size*2, 2)
        self.bn = nn.BatchNorm1d(2)
        self.act = nn.ReLU()

    def forward(self,g1,g2):
        x1, edge_index1, batch1 = g1.x, g1.edge_index,g1.batch
        x2, edge_index2, batch2 = g2.x, g2.edge_index,g2.batch
        
        x1 = self.fcl(x1);x2 = self.fcr(x2)
        out1 = self.act(self.conv1(x1, edge_index1))
        out2 = self.act(self.conv1(x2, edge_index2))
        z1 = self.conv2(out1, edge_index1)
        z2 = self.conv2(out2, edge_index2)

        z1 = global_mean_pool(z1, batch1)
        z2 = global_mean_pool(z2, batch2)

        x = torch.cat([z1, z2], dim=1)    #[batch_size,out_size*2]
        out = self.fc1(x)                      #[batch_size,in_feats]
        return 0,out,0,0


class GNN(nn.Module):
    def __init__(self, args, in_feats, hidden_size,out_size):
        super(GNN, self).__init__()
        self.args = args
        self.in_feats =in_feats
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.Mish = nn.Mish()
        self.bn = nn.BatchNorm1d(out_size)

        self.fc = nn.Linear(hidden_size, out_size)
        self.fc1 = nn.Linear(out_size, 2)

    def forward(self, x, edge_index,batch):
        out = self.Mish(self.conv1(x, edge_index))
        out = self.Mish(self.conv2(out, edge_index))      #[num_node*32,hidden_size]
        x1 = out.clone()
        x1 = self.Mish(self.fc(x1))                #[num_node*32,out_size] 
        x2 = global_mean_pool(x1, batch)    #[32,out_size]
        x2 = self.bn(x2)
        return x1,x2
        
class Attention(nn.Module):
    def __init__(self,arg, in_feats,out_size,multi_numROI):
        super(Attention, self).__init__()
        self.arg = arg
        self.in_feats = in_feats
        self.out_size = out_size
        self.multi_numROI = multi_numROI
        self.query = torch.nn.Linear(in_feats, out_size)
        self.key = torch.nn.Linear(in_feats, out_size)
        self.value = torch.nn.Linear(in_feats, out_size)

        self.classifier = torch.nn.Linear(in_feats, out_size)
        self.dropout = nn.Dropout(p=arg.dropout_ratio)

    def forward(self, x,num_graphs):
        x_change = x.reshape(num_graphs,self.multi_numROI, self.in_feats)
        Q = self.query(x_change)
        K = self.key(x_change)
        V = self.value(x_change)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = torch.matmul(attention_weights, V)
        x = out.reshape(self.multi_numROI * num_graphs, self.out_size)
        attention_weights = torch.mean(attention_weights, dim=0)
        return x,attention_weights

class KnowMDD(torch.nn.Module):
    def __init__(self, args, in_feats, hidden_size,out_size):
        super(KnowMDD,self).__init__()
        self.args = args
        self.in_feats =in_feats
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.multi_numROI = args.multi_numROI
        self.keep_nodes = args.multi_keep

        self.fcl = nn.Linear(self.multi_numROI[0]+3, in_feats)
        self.fcr = nn.Linear(self.multi_numROI[1]+3, in_feats)
        self.fcl1 = nn.Linear(self.multi_numROI[0], in_feats)
        self.fcr1 = nn.Linear(self.multi_numROI[1], in_feats)

        self.conv = GCNConv(in_feats, hidden_size)
        self.conv1 = GCNConv(hidden_size, hidden_size)
        self.mlp = nn.Linear(hidden_size,out_size)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Mish(),
        #     nn.Linear(hidden_size, out_size)
        # )

        self.Attention1 = Attention(args,in_feats,out_size,self.multi_numROI[0])
        self.Attention2 = Attention(args,in_feats,out_size,self.multi_numROI[1])
        self.fcattention = nn.Linear(in_feats,out_size)      
        self.conv4 = GCNConv(hidden_size, hidden_size)
        self.Mish = nn.Mish()
        
        self.fc1 = nn.Sequential(
            nn.Linear(out_size*4, in_feats),
            nn.Mish(),
            nn.Linear(in_feats, 2),
            # nn.BatchNorm1d(out_size)
        )

        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=args.dropout_ratio)

    def forward(self,g1,g2):
        x1, edge_index1, batch1,num_graphs1 = g1.x, g1.edge_index,g1.batch,g1.num_graphs
        x2, edge_index2, batch2,num_graphs2 = g2.x, g2.edge_index,g2.batch,g2.num_graphs
        z1 = self.fcl1(x1[:,:-3]);z2 = self.fcr1(x2[:,:-3])
        z1,attention_weights1 = self.Attention1(z1,num_graphs1)
        z2,attention_weights2 = self.Attention2(z2,num_graphs2)
        
        z1 = global_mean_pool(z1, batch1)
        z2 = global_mean_pool(z2, batch2)   #global_mean_pool

        x1 = self.fcl(x1);x2 = self.fcr(x2)
        out1 = self.Mish(self.conv(x1, edge_index1))
        out2 = self.Mish(self.conv(x2, edge_index2))
        out1 = self.dropout(out1); out2 = self.dropout(out2)
        out1 = self.conv1(out1, edge_index1)
        out2 = self.conv1(out2, edge_index2)
        
        #h1/h2
        h1,sub_batch1 = self.RW_subgraph(out1,g1,self.keep_nodes[0],self.args.sub_graph)
        h2,sub_batch2 = self.RW_subgraph(out2,g2,self.keep_nodes[1],self.args.sub_graph)
        h1 = self.mlp(h1)
        h2 = self.mlp(h2)
        # h1 = self.mlp(out1)
        # h2 = self.mlp(out2)
        
        # h1 = global_mean_pool(h1, batch1)
        # h2 = global_mean_pool(h2, batch2)
        h1 = global_mean_pool(h1, sub_batch1)
        h2 = global_mean_pool(h2, sub_batch2)


        c_loss = contrastive_infoloss(h1,z2)+contrastive_infoloss(z1,h2)
        # c_loss = 0

        x = torch.cat([h1, z1,h2,z2], dim=1)    #[batch_size,out_size*2]
        out = self.fc1(x)                      #[batch_size,in_feats]
        return c_loss,out,attention_weights1,attention_weights2
    def RW_subgraph(self,h,data,start_list,walk_length):
        edge_index = data.edge_index
        batch = data.batch
        all_visited_nodes = set()
        start_nodes = []
        start_nodes = torch.tensor(start_list, dtype=torch.long)

        for start_node in start_nodes:
            walk = [start_node]
            choice_node = {start_node}
            current_node = start_node
            for _ in range(walk_length - 1):
                neighbors = torch.cat([edge_index[1][edge_index[0] == current_node]]).tolist()
                neighbors = [node for node in neighbors if node not in choice_node]
                if not neighbors:
                    break
                current_node = random.choice(neighbors)
                walk.append(current_node)
                choice_node.add(current_node)
            all_visited_nodes.update(choice_node)
        all_visited_nodes = torch.tensor(list(all_visited_nodes))
        visited_nodes = []
        for i in range(data.num_graphs):
            visited_nodes.extend([node + i * h.size(0) // data.num_graphs for node in all_visited_nodes])
        subgraph_x = h[visited_nodes]
        subgraph_batch = batch[visited_nodes]
        return subgraph_x,subgraph_batch
def contrastive_loss(h, z, temperature=0.6):
    batch_size = h.size(0)
    h = F.normalize(h, dim=1)
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.mm(h, z.t())  # (batch_size, batch_size)
    similarity_matrix /= temperature
    
    positive_samples = torch.diag(similarity_matrix)
    logsumexp = torch.logsumexp(similarity_matrix, dim=1)

    loss = -positive_samples + logsumexp
    loss = loss.mean()
    
    return loss

def contrastive_infoloss(x1, x2, temperature=0.6):
    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)
    logits = torch.matmul(x1, x2.t()) / temperature
    labels = torch.eye(x1.shape[0], dtype=torch.float, device=x1.device)
    loss = F.cross_entropy(logits, labels)
    return loss
