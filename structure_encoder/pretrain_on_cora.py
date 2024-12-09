import os

from gpse import GPSE
from gpse_mlp import GPSE_MLP
import torch
from torch import nn, optim
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
# from get_cora_subgraphs import CustomDataset
import networkx as nx

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora, Arxiv, Pubmed



def sim(z1, z2):
    z1 = F.normalize(z1) 
    z2 = F.normalize(z2) 
    return torch.mm(z1, z2.t()) 

def get_node_sim_matrix(node_features):
    return sim(node_features, node_features)

def struct_text_loss(batch, node_sims, criterion, i):
    struct_sims = sim(batch, batch)
    if (i+1)%50 == 0:
        print(f'  struct_sims of node 0 and node 1-10:')
        print(" ", struct_sims[0][1:11])
        print(f'  text_sims of node 0 and node 1-10:')
        print(" ", node_sims[0][1:11])
        print(f'  struct_sims of node 0 and its neighbors:')
        print(" ", float(struct_sims[0][1184]),\
                float(struct_sims[0][1207]),\
                float(struct_sims[0][1408]),\
                float(struct_sims[0][1626]), \
                float(struct_sims[0][2414]))
        print(f'  text_sims of node 0 and its neighbors:')
        print(" ", float(node_sims[0][1184]),\
                float(node_sims[0][1207]),\
                float(node_sims[0][1408]),\
                float(node_sims[0][1626]), \
                float(node_sims[0][2414]))
    loss = criterion(struct_sims, node_sims)
    return loss

def constractive_loss(batch, A, t):
    f = lambda x: torch.exp(x/t) 
    loss = 0
    sim_m = f(sim(batch, batch))
    neg_sim = (sim_m.sum(0) - sim_m.diag()).sum()
    pos_sim = torch.sum(sim_m[A==1])
    loss = -torch.log(pos_sim/neg_sim)
    return loss/len(batch)

def compute_k_neigh_sim(A, sims, k):  
    num_nodes = A.shape[0]     
    for i in range(1, k+1):
        A_k = np.linalg.matrix_power(A, i)
        avg_sims = torch.mean(sims[A_k==1])
        print(f"Average similarity for {i}-hop neighbors: {float(avg_sims):.4f}")


def train(module, dataset, node_sims, epochs, device):
    module.to(device)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    criterion = nn.MSELoss().to(device)
    data = dataset[0].to(device)
    num_nodes = len(data.x)
    A = torch.zeros((num_nodes, num_nodes))  
    A[data.edge_index[0], data.edge_index[1]] = 1  
    A[data.edge_index[1], data.edge_index[0]] = 1
    rand = np.random.normal(loc=0, scale=1.0, size=(len(data.x), 20))
    data.x = torch.from_numpy(rand.astype('float32')).to(device)
    data.x[-1] = 0

    for i in range(epochs):
        output = module(data) 
        s_loss = struct_text_loss(output, node_sims, criterion, i)
        c_loss = module.constractive_loss(data, output, 0.2)
        loss = s_loss*0 + c_loss
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        # print(f'sg {j}, loss: {loss.item():.6f}')
        if (i+1)%50 == 0:
            compute_k_neigh_sim(A, sim(output, output), 3)
            print('-----------------------------------------------------------------------------------------------')
            print(f'epoch {i} loss: {loss.item():.6f}, s_loss = {s_loss.item():.6f}, c_loss = {c_loss.item():.6f}')
            print('-----------------------------------------------------------------------------------------------')


def eval(module, dataset, device, node_sims=None):
    # module.to(device)
    # module.eval()
    # pretrain_m_dim_in = 20
    # data = dataset[0].to(device)
    # rand = np.random.normal(loc=0, scale=1.0, size=(len(data.x), pretrain_m_dim_in))
    # data.x = torch.from_numpy(rand.astype('float32')).to(device)
    # data.x[-1] = 0
    # output = module(data)
    # # torch.save(output, './output_cora_tag_pt_module.pt')

    data = dataset[0]
    num_nodes = len(data.x)
    A = torch.zeros((num_nodes, num_nodes))  
    A[data.edge_index[0], data.edge_index[1]] = 1  
    A[data.edge_index[1], data.edge_index[0]] = 1
    # sims = sim(output, output)

    texts = torch.load('/home/wangjingchu/code/SDGLM/TAGDataset/pubmed/task/ST/node_features.pt')
    node_sims = sim(texts, texts)

    print('=================================================')
    print(f'structure repersentation sims:')
    # compute_k_neigh_sim(A, sims, 5)
    print('=================================================')
    print(f'text repersentation sims:')
    if node_sims is not None:
        compute_k_neigh_sim(A, node_sims, 5)

    # subgraph_nodes = torch.load('/home/wangjingchu/code/SDGLM/TAGDataset/cora/subgraph_nodes.pt')
    # print(subgraph_nodes[0])
    # subgraph_sims_0 = [float(sims[0, j]) for j in subgraph_nodes[0]]
    # print(subgraph_sims_0)
    # subgraph_text_sims_0 = [float(node_sims[0, j]) for j in subgraph_nodes[0]]
    # print(subgraph_text_sims_0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    dataset = Cora()
    node_features = torch.load('./TAGDataset/cora/task/ST/node_features.pt').to(device)
    node_sims = get_node_sim_matrix(node_features)
    gpse = GPSE.from_pretrained('molpcba').to(device)
    module = GPSE_MLP(gpse, 512, 1024, 1024, 4).to(device)
    print('-----------------------------')
    train(module, dataset, node_sims, 500, device)
    torch.save(module, './cora_tag_pt_module.pt')
    module = torch.load('./cora_tag_pt_module.pt')
    dataset = Pubmed()
    eval(module, dataset, device)

if __name__ == '__main__':
    main()