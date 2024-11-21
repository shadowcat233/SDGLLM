import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
import os
from gpse import precompute_GPSE, GPSE
from torch_geometric.datasets import Planetoid


class CustomDataset(Dataset):
    def __init__(self, subgraphs):
        super(CustomDataset, self).__init__()
        self.subgraphs = subgraphs
 
    def len(self):
        return len(self.subgraphs)
 
    def get(self, idx):
        subgraph_data = self.subgraphs[idx]
        graph = subgraph_data['graph']
        nodes = subgraph_data['nodes']
        x = subgraph_data['x']
        edge_index = subgraph_data['edge_index']
        y = subgraph_data.get('y', None) 
        data = Data(x=x, edge_index=edge_index)
        if y is not None:
            data.y = y
        data.A = subgraph_data['A']
        data.center = subgraph_data['center']
        return data



def limit_k_order_neighbors(G, center_node, depth, max_neighbors):
    result_G = nx.Graph()
    visited = set()
    queue = [(center_node, 0)]  # (node, distance_from_center)
    while queue:
        current_node, distance = queue.pop(0)
        if distance >= depth:
            continue
        if current_node not in visited:
            visited.add(current_node)
            result_G.add_node(current_node)
            neighbors = list(G.neighbors(current_node))
            if len(neighbors) > max_neighbors:
                neighbors_to_keep = np.random.choice(neighbors, max_neighbors, replace=False)
            else:
                neighbors_to_keep = neighbors
            for neighbor in neighbors_to_keep:
                result_G.add_edge(current_node, neighbor)
                if neighbor not in visited and distance < depth:
                    queue.append((neighbor, distance + 1))
    return result_G

def main():
    dataset = Planetoid('./', "Cora")
    data = dataset[0]
    G = nx.Graph()
    for i in range(data.x.size(0)): G.add_node(i)
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[0][i], data.edge_index[1][i]
        G.add_edge(int(u), int(v))
        G.add_edge(int(v), int(u))

    subgraphs = []
    for node in G.nodes():
        subgraph = limit_k_order_neighbors(G, node, depth=3, max_neighbors=5)
        nodes = list(subgraph.nodes())
        edges = list(subgraph.edges())
        node_index_map = {n: i for i, n in enumerate(nodes)}
        edge_index = torch.tensor([[node_index_map[u] for u, v in edges],
                                [node_index_map[v] for u, v in edges]], dtype=torch.long)
        num_nodes = len(nodes)
        A = torch.zeros((num_nodes, num_nodes))  
        A[edge_index[0], edge_index[1]] = 1  
        A[edge_index[1], edge_index[0]] = 1
        subgraph_data = {
            'graph': subgraph,
            'nodes': nodes,
            'center': node_index_map[node],
            'x': data.x[nodes],  
            'edge_index': edge_index, 
            'y': data.y[nodes],
            'A': A
        }
        subgraphs.append(subgraph_data)

        
    dataset = CustomDataset(subgraphs)
    torch.save(dataset, './Cora/subgraphs.pt')

if __name__ == '__main__':
    main()