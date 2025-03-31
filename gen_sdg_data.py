from TAGLAS.datasets import Cora, Pubmed, Arxiv, Products, WikiCS
from TAGLAS.tasks import SubgraphTextNPTask
from TAGLAS.tasks.text_encoder import SentenceEncoder
import torch
from tqdm import tqdm
import random

dts = [Cora, Pubmed, Arxiv, WikiCS, Products]
names = ['cora', 'pubmed', 'arxiv', 'wikics', 'products']
modes = ['node', 'edge']

num = 4
n_mode = 0
mode = modes[n_mode]
name = names[num]
dataset = dts[num]()
data = dataset[0]

print(data)
print(data.label, len(data.label))

struct_encodes = None #torch.load('/home/wangjingchu/code/SDGLM/structure_encoder/output_pubmed_tag_pt_module.pt')
struct_encodes = None #torch.load('/home/wangjingchu/code/SDGLM/structure_encoder/output_pubmed_tag_pt_module.pt')

# task = SubgraphTextNPTask(dataset)
# task.hop = 1

# subgraph_nodes = []
# subgraph_edge_index = []

# for i in tqdm(range(len(data.node_map)), desc="Processing nodes", unit="node"):

#     processed_edge_index, processed_node_map, _, _ = \
#         task.__process_graph__(i, dataset[0].edge_index, dataset[0].node_map, dataset[0].edge_map)

#     processed_edge_index = processed_node_map[processed_edge_index]

#     indices = (processed_node_map == i).nonzero(as_tuple=True)[0]
#     processed_node_map[indices[0]] = processed_node_map[0]
#     processed_node_map[0] = i
#     if len(processed_node_map)>10:
#         random.shuffle(processed_node_map)
#         processed_node_map = processed_node_map[:10]
#     subgraph_nodes.append(processed_node_map.tolist())

#     mask_0 = torch.isin(processed_edge_index[0], processed_node_map)
#     mask_1 = torch.isin(processed_edge_index[1], processed_node_map)

#     mask = mask_0 & mask_1
#     processed_edge_index = processed_edge_index[:, mask]
#     subgraph_edge_index.append(processed_edge_index.tolist())

# # print(subgraph_nodes[:10], subgraph_edge_index[:10])

# torch.save(subgraph_nodes, f'./TAGDataset/{name}/subgraph_nodes_h1.pt')
# torch.save(subgraph_edge_index, f'./TAGDataset/{name}/subgraph_edge_index_h1.pt')
# # subgraph_nodes = torch.load('./TAGDataset/cora/subgraph_nodes.pt')
subgraph_nodes = torch.load(f'./TAGDataset/{name}/subgraph_nodes_h1.pt')
subgraph_edge_index = torch.load(f'./TAGDataset/{name}/subgraph_edge_index_h1.pt')

# print(subgraph_nodes[0])
# print(subgraph_edge_index[0])
# edge_index = []
# for i in range(len(subgraph_nodes)):
#     subgraph_nodes[i] = subgraph_nodes[i][:8]
#     sgn = torch.tensor(subgraph_nodes[i], dtype=torch.long)
#     edge_index_i = torch.tensor(subgraph_edge_index[i], dtype=torch.long)
#     mask_0 = torch.isin(edge_index_i[0], sgn)
#     mask_1 = torch.isin(edge_index_i[1], sgn)
#     mask = mask_0 & mask_1
#     processed_edge_index = edge_index_i[:, mask]
#     edge_index.append(processed_edge_index.tolist())

# subgraph_edge_index = edge_index
# print(subgraph_nodes[0])
# print(subgraph_edge_index[0])

# def divide_nodes_by_subgraphs(subgraph_nodes, subgraph_edge_index, start, end, threshold=1, visited=None):
#     """
#     将所有节点划分为多个集合，每个集合满足：其中的节点的所有子图节点都在该集合中。
#     :param subgraph_nodes: List[List[int]]，每个节点的三阶子图节点的列表
#     :param threshold: int，每个集合的满足条件的节点数量阈值
#     :return: List[List[int]]，划分后的集合列表
#     """
#     num_nodes = len(subgraph_nodes)  # 总节点数
#     visited = [False] * num_nodes if visited is None else visited # 跟踪所有节点是否已分配到某个集合
#     sets = []  # 存储结果集合列表
#     valids = []
#     e_idxs = []
#     valid_batchs = {}

#     def expand_set(start_node, threshold):
#         """
#         从 start_node 开始扩展集合，直到达到阈值或满足条件
#         """
#         current_set = set()
#         edge_set = set()
#         valid_nodes = []
#         queue = [start_node]
#         while queue:
#             node = queue.pop(0)
#             edges = subgraph_edge_index[node]
#             edges_tuple = [(edges[0][i], edges[1][i]) for i in range(len(edges[0]))]
#             for edge in edges_tuple: edge_set.add(edge)
#             if node not in current_set:
#                 current_set.add(node)
#                 if visited[node]: continue
#                 for neighbor in subgraph_nodes[node]:
#                     if neighbor not in current_set:
#                         queue.append(neighbor)

#             # 如果达到阈值，检查集合是否满足所有子图节点都在集合中
#             if len(current_set) >= threshold:
#                 valid_nodes = [n for n in current_set if all(m in current_set for m in subgraph_nodes[n]) and visited[n] is False and data.label[data.label_map[n]]!='MISSING']
#                 if len(valid_nodes) >= threshold:
#                     break

#         edge_set = {edge for edge in edge_set if edge[0] in current_set and edge[1] in current_set}
#         e_idx = [[e[0] for e in edge_set], [e[1] for e in edge_set]]
#         valid_nodes = [n for n in current_set if all(m in current_set for m in subgraph_nodes[n]) and visited[n] is False] # and data.label[data.label_map[n]]!='MISSING']

#         return current_set, valid_nodes, edge_set

#     for start_node in range(start, end):
#         if len(subgraph_nodes[start_node])==1: continue
#         if not visited[start_node]:
#             # 扩展集合
#             new_set, new_valid, edge_set = expand_set(start_node, threshold)      

            for node in new_valid:
                visited[node] = True
            if len(new_valid) < 0.5*threshold:
                # 合并到最后一个集合
                if sets and len(valids[-1]) + len(new_valid) <= threshold * 1.5:
                    sets[-1] = list(set(sets[-1]) | new_set)
                    valids[-1] = list(set(valids[-1]) | set(new_valid))
                    pe = e_idxs[-1]
                    pe_tuple = set([(pe[0][i].item(), pe[1][i].item()) for i in range(len(pe[0]))])
                    new_e_set = pe_tuple | edge_set
                    e_idx = [[e[0] for e in edge_set], [e[1] for e in new_e_set]]
                    e_idxs[-1] = e_idx
            else:
                # 将集合转换为列表并添加到结果中
                sets.append(list(new_set))
                valids.append(new_valid)
                e_idx = [[e[0] for e in edge_set], [e[1] for e in edge_set]]
                e_idxs.append(e_idx)
                
    return sets, valids, e_idxs

dlen = len(data.x)
batchs, valids, edges = divide_nodes_by_subgraphs(subgraph_nodes, subgraph_edge_index, 0, int(0.8*dlen))
split = len(batchs)
print(split)
print(edges[0])
b2, v2, e2 = divide_nodes_by_subgraphs(subgraph_nodes, subgraph_edge_index, int(0.8*dlen), 2708)
batchs = batchs + b2
valids = valids + v2
edges = edges + e2
b_f_idx = [b[0] for b in batchs]
print(b_f_idx)
print('------------------------')
# print(len(batchs))
# print('------------------------')
# for i in range(len(batchs)):
#     print(len(batchs[i]), len(valids[i]))
# print('------------------------')
# print(batchs)
# print('------------------------')
# print(valids)
# torch.save(batchs, f'./TAGDataset/{name}/batchs.pt')
# torch.save(valids, f'./TAGDataset/{name}/valids.pt')
# torch.save(edges, f'./TAGDataset/{name}/edges.pt')
batchs = torch.load(f'./TAGDataset/{name}/batchs.pt')
valids = torch.load(f'./TAGDataset/{name}/valids.pt')
edges = torch.load(f'./TAGDataset/{name}/edges.pt')

valid_nodes_masks = [
        [1 if node in valids[i] else 0 for node in batchs[i]]
        for i in range(len(batchs))
    ]

from sdg_dataset import SDGDataset, SDGEdgeDataset

from transformers import AutoTokenizer

model_path = './llm/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)

objects = ['paper', 'paper', 'paper', 'website', 'product']
obj = objects[num]

if mode == 'node':
    categories = [lab for lab in data.label if lab not in {'Yes', 'No', 'MISSING'}]
    categories_str = ', '.join(map(str, categories))
    inst = {}
    inst['head'] = f"<<SYS>>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful\
    answers to the user's questions.\n<<\SYS>>\
    <s>[INST] Here's the title and abstract or content of a {obj}, please tell me which category it belongs to.\n"
    inst['tail'] = f"\nOptional Categories: {categories_str}\n\
    Please select one of the options from the above list. \
    Don't add any other replies.[\INST] \nThe {obj} belongs to the category of "

    print(inst)

    true_labels = [data.label[data.label_map[i]] for i in range(len(data.x))]

    # struct_encodes = torch.load("./structure_encoder/output_cora_tag_pt_module.pt").to('cpu')

cora_sdg_dts = SDGDataset(data.x, true_labels, struct_encodes, batchs, subgraph_nodes, edges, valid_nodes_masks, tokenizer, inst, split)
batch = cora_sdg_dts[0]
print(batch)

torch.save(cora_sdg_dts, './pubmed_sdg_dataset.pt')

