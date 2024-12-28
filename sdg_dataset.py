from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
from torch_geometric.data import Data
import numpy as np
import random

def select_samples(dataset, num_samples):
    """
    从一个 SDGDataset 中随机选取 num_samples 条数据。
    """
    indices = (range(num_samples))
    return [dataset[i] for i in indices]

class MergedSDGDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, num_samples_a, num_samples_b):
        """
        初始化合并数据集，从两个数据集中各选取指定数量的数据。
        """
        self.samples = select_samples(dataset_a, num_samples_a) + select_samples(dataset_b, num_samples_b)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SDGDataset(Dataset):
    def __init__(self, texts, labels, struct_encodes, batchs, subgraphs, edges, valid_nodes_masks, tokenizer, inst, split, weight=None):
        self.struct_encodes = struct_encodes
        self.subgraphs = subgraphs
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.batchs = batchs
        self.valid_nodes_masks = valid_nodes_masks
        self.split = split
        self.dtype = torch.float16
        self.edges = edges
        self.weight = weight

        self.texts_ids = self.tokenizer(
            texts,
            max_length=130,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][:, 1:]
        # i = 0
        # for ids in self.texts_ids:
        #     if ids[-1] == 0: i = i + 1
        # print(f'last token is padding rate: {i/2708}')
        self.inst_head_ids = self.tokenizer(
            inst['head'],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)[1:]
        self.inst_tail_ids = self.tokenizer(
            inst['tail'],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)[1:]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_ids = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][:, 1:]
        end_token_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]
        end_token_tensor = end_token_id * torch.ones(self.label_ids.size(0), 1, dtype=torch.long)
        self.label_ids = torch.cat([self.label_ids, end_token_tensor], dim=1)
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def set_dtype(self, dtype):
        self.dtype = dtype
        

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        start = len(self.inst_head_ids)
        end = start + len(self.texts_ids[0])
        t_ids = torch.stack([
            torch.cat([self.inst_head_ids, self.texts_ids[i], self.inst_tail_ids, self.label_ids[i]], dim=0)
            for i in self.batchs[idx]
        ])
        node_info_mask = torch.zeros(len(t_ids[0]), dtype=torch.bool)
        node_info_mask[start:end] = True

        attention_mask = torch.ones((t_ids.size(0), t_ids.size(1)))
        attention_mask[t_ids==self.tokenizer.pad_token_id] = 0

        inv = {self.batchs[idx][i]: i for i in range(len(self.batchs[idx]))}

        sg_nodes = torch.zeros(len(self.batchs[idx]), len(self.batchs[idx]))
        for i in range(len(self.batchs[idx])):
            for neigh in self.subgraphs[self.batchs[idx][i]]:
                if neigh in self.batchs[idx]:
                    sg_nodes[inv[self.batchs[idx][i]], inv[neigh]] = 1

        struct_encode = self.struct_encodes[self.batchs[idx]]

        edges = self.edges[idx]
        edges = [[inv[i] for i in e] for e in edges]
        edges = torch.tensor(edges, dtype=torch.long)

        rand = np.random.normal(loc=0, scale=1.0, size=(len(self.batchs[idx]), 20))
        x = torch.from_numpy(rand.astype('float32'))
        x[-1] = 0
        graph = Data(x=x, edge_index=edges)

        # if self.weight is not None:
        #     sims = self.weight[self.batchs[idx]][:, self.batchs[idx]]
        # else: sims = None

        return {
            "input_ids": t_ids,
            "attention_mask": attention_mask,
            "struct_encode": struct_encode,
            "subgraph_nodes": sg_nodes,
            "graph": graph,
            "valid_nodes_mask": torch.tensor(self.valid_nodes_masks[idx]),
            "labels": t_ids,
            "node_info_mask": node_info_mask
            # "sims": sims
        }


if __name__ == "__main__":
    cora = torch.load('./cora_sdg_dataset.pt')
    data = cora[0]
    print(cora.struct_encodes.shape)
    print(data['attention_mask'][0])