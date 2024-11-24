from TAGLAS.datasets import Cora
from TAGLAS.tasks import SubgraphTextNPTask
from TAGLAS.tasks.text_encoder import SentenceEncoder
import torch

dataset = Cora()
data = dataset[0]

# print(data.x[0])
# print(data.label)

# task = SubgraphTextNPTask(dataset)

# subgraph_nodes = []
# for i in range(len(data.node_map)):
#     _, processed_node_map, _, _ = \
#         task.__process_graph__(i, dataset[0].edge_index, dataset[0].node_map, dataset[0].edge_map)
#     indices = (processed_node_map == i).nonzero(as_tuple=True)[0]
#     processed_node_map[indices[0]] = processed_node_map[0]
#     processed_node_map[0] = i
#     subgraph_nodes.append(processed_node_map.tolist())

# print(subgraph_nodes[1000:1010])


# torch.save(subgraph_nodes, './TAGDataset/cora/subgraph_nodes.pt')
subgraph_nodes = torch.load('./TAGDataset/cora/subgraph_nodes.pt')

def divide_nodes_by_subgraphs(subgraph_nodes, start, end, threshold=4):
    """
    将所有节点划分为多个集合，每个集合满足：其中的节点的所有子图节点都在该集合中。
    :param subgraph_nodes: List[List[int]]，每个节点的三阶子图节点的列表
    :param threshold: int，每个集合的满足条件的节点数量阈值
    :return: List[List[int]]，划分后的集合列表
    """
    num_nodes = len(subgraph_nodes)  # 总节点数
    visited = [False] * num_nodes  # 跟踪所有节点是否已分配到某个集合
    sets = []  # 存储结果集合列表
    valids = []

    def expand_set(start_node, threshold):
        """
        从 start_node 开始扩展集合，直到达到阈值或满足条件
        """
        current_set = set()
        valid_nodes = []
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node not in current_set:
                current_set.add(node)
                if visited[node]: continue
                for neighbor in subgraph_nodes[node]:
                    if neighbor not in current_set:
                        queue.append(neighbor)

            # 如果达到阈值，检查集合是否满足所有子图节点都在集合中
            if len(current_set) >= threshold:
                valid_nodes = [n for n in current_set if all(m in current_set for m in subgraph_nodes[n])]
                if len(valid_nodes) >= threshold:
                    break

        valid_nodes = [n for n in current_set if all(m in current_set for m in subgraph_nodes[n])]

        return current_set, valid_nodes

    for start_node in range(start, end):
        if not visited[start_node]:
            # 扩展集合
            new_set, new_valid = expand_set(start_node, threshold)
            for node in new_valid:
                visited[node] = True
            if len(new_valid) < 3: #threshold/2:
                # 合并到最后一个集合
                if sets and len(valids[-1]) + len(new_valid) <= threshold * 1.5:
                    sets[-1] = list(set(sets[-1]) | new_set)
                    valids[-1] = list(set(valids[-1]) | set(new_valid))
            else:
                # 将集合转换为列表并添加到结果中
                sets.append(list(new_set))
                valids.append(new_valid)
                
    return sets, valids

batchs, valids = divide_nodes_by_subgraphs(subgraph_nodes, 0, 2216)
split = len(batchs)
print(split)
b2, v2 = divide_nodes_by_subgraphs(subgraph_nodes, 2217, 2708)
batchs = batchs + b2
valids = valids + v2
print('------------------------')
print(len(batchs))
print('------------------------')
for i in range(len(batchs)):
    print(len(batchs[i]), len(valids[i]))
# print('------------------------')
# print(batchs)
# print('------------------------')
# print(valids)
torch.save(batchs, './TAGDataset/cora/batchs.pt')
torch.save(valids, './TAGDataset/cora/valids.pt')

from sdg_dataset import SDGDataset

from transformers import AutoTokenizer

model_path = '/home/wangjingchu/code/SDGLM/llm/Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)

inst = {}
inst['head'] = "<<SYS>>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, \
and polite answers to the user's questions.\n<<\SYS>>\
<s>[INST] Here's the title and abstruct of a paper, please tell me which category the paper belongs to.\n"
inst['tail'] = "Optional Categories: Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods\n\
Please select one of the options from the above list that is the most likely category. \
Only answer the name of the category and don't add any other replies.[\INST] The paper belongs to the category of "

valid_nodes_masks = [
    [1 if node in valids[i] else 0 for node in batchs[i]]
    for i in range(len(batchs))
]

true_labels = [data.label[data.label_map[i]] for i in range(len(data.x))]

struct_encodes = torch.load("/home/wangjingchu/code/SDGLM/structure_encoder/output_cora_tag_pt_module.pt").to('cpu')

cora_sdg_dts = SDGDataset(data.x, true_labels, struct_encodes, batchs, subgraph_nodes, valid_nodes_masks, tokenizer, inst, split)
batch = cora_sdg_dts[0]
print(batch["input_ids"][0])

torch.save(cora_sdg_dts, './cora_sdg_dataset.pt')

