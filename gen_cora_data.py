from TAGLAS.datasets import Cora
from TAGLAS.tasks import SubgraphTextNPTask
from TAGLAS.tasks.text_encoder import SentenceEncoder
dataset = Cora()
data = dataset[0]

print(data.x[0])
print(data.label[data.label_map[0]])

task = SubgraphTextNPTask(dataset)

subgraph_nodes = []
for i in range(len(data.node_map)):
    _, processed_node_map, _, _ = \
        task.__process_graph__(i, dataset[0].edge_index, dataset[0].node_map, dataset[0].edge_map)
    indices = (processed_node_map == i).nonzero(as_tuple=True)[0]
    processed_node_map[indices[0]] = processed_node_map[0]
    processed_node_map[0] = i
    subgraph_nodes.append(processed_node_map.tolist())

print(subgraph_nodes[1000:1010])

import torch
torch.save(subgraph_nodes, './TAGDataset/cora/subgraph_nodes.pt')
