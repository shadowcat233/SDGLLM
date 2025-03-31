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

def tokens2str(tensor):
    tensor_list = tensor.tolist()
    str_list = [str(i) for i in tensor_list]
    return '-'.join(str_list)

class MergedSDGDataset(Dataset):

    def __init__(self):
        self.samples = []

    def merge_init(self, datasets, num_samples, init=True):
        """
        初始化合并数据集，从两个数据集中各选取指定数量的数据。
        """
        for i in range(len(datasets)):
            self.samples = self.samples + select_samples(datasets[i], num_samples[i])
        random.shuffle(self.samples)
    
    def few_shot_init(self, dataset, n):
        ctgr = [tokens2str(dataset.label_ids[i]) for i in range(len(dataset.label_ids))]
        ctgr = set(ctgr)
        dic = {c: 0 for c in ctgr}
        lab_len = len(dataset.label_ids[0])
        for data in dataset:
            valid_node_mask = data['valid_nodes_mask']
            if valid_node_mask.sum()>1: continue
            lab = data['input_ids'][valid_node_mask==1][0][-lab_len:]
            lab = tokens2str(lab)
            if dic[lab]<n:
                dic[lab] += 1
                self.samples.append(data)
                if all(cnt >= n for cnt in dic.values()):
                    break
            else: continue
        v = [cnt for cnt in dic.values()]
        print(v, len(v))

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
        self.split = split
        self.dtype = torch.float16
        self.edges = edges
        self.weight = weight
        self.valid_nodes_masks = valid_nodes_masks

        self.texts_ids = self._tokenize(texts, 120)[:, 1:]
        self.inst_head_ids = self._tokenize(inst['head']).squeeze(0)[1:]
        self.inst_tail_ids = self._tokenize(inst['tail']).squeeze(0)[1:]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_ids = self._tokenize(labels)[:, 1:]
        end_token_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]
        end_token_tensor = end_token_id * torch.ones(self.label_ids.size(0), 1, dtype=torch.long)
        self.label_ids = torch.cat([self.label_ids, end_token_tensor], dim=1)
        self.tokenizer.pad_token = self.tokenizer.unk_token

    def set_dtype(self, dtype):
        self.dtype = dtype

    def _tokenize(self, texts, max_length=None):
        """分词处理的辅助方法"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]  

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

        struct_encode = None if self.struct_encodes is None else self.struct_encodes[self.batchs[idx]]

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
            "x": x,
            "edge_index": edges,
            "edge_index2": None,
            "valid_nodes_mask": torch.tensor(self.valid_nodes_masks[idx]),
            "labels": t_ids,
            "node_info_mask": node_info_mask,
            "mode": 0
        }

class SDGEdgeDataset(Dataset):
    def __init__(self, texts, labels, struct_encodes, batchs, subgraphs, edges, valid_batchs, tokenizer, inst, split, edge_index):
        self.struct_encodes = struct_encodes
        self.subgraphs = subgraphs
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.batchs = batchs
        self.valid_batchs = valid_batchs
        self.split = split
        self.dtype = torch.float16
        self.edges = edges

        # 分词处理
        self.texts_ids = self._tokenize(texts, 120)[:, 1:]
        self.inst_head_ids = self._tokenize(inst['head']).squeeze(0)[1:]
        self.inst_mid_ids = self._tokenize(inst['mid']).squeeze(0)[1:]
        self.inst_tail_ids = self._tokenize(inst['tail']).squeeze(0)[1:]

        # 处理标签
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_ids = self._tokenize(labels)[:, 1:]
        end_token_id = self.tokenizer.encode(self.tokenizer.eos_token, add_special_tokens=False)[0]
        end_token_tensor = end_token_id * torch.ones(self.label_ids.size(0), 1, dtype=torch.long)
        self.label_ids = torch.cat([self.label_ids, end_token_tensor], dim=1)
        self.tokenizer.pad_token = self.tokenizer.unk_token

        # 构建边集合
        # self.edge_index = edge_index
        self.edge_set = self._build_edge_set(edge_index[:, :split])
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)
        self.edge_index = edge_index[:, perm]

    def _append_edges(self):
        elen = len(self.edges)
        for b in self.batchs[elen:] :
            i = b[0]
            self.edges.append([[i], [i]])

    def _valid_batchs(self):
        self.valid_batchs = {int(k): v for k, v in self.valid_batchs.items()}

    def _tokenize(self, texts, max_length=None):
        """分词处理的辅助方法"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]

    def _build_edge_set(self, edge_index):
        """构建边集合的辅助方法"""
        edge_set = set()
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            edge_set.add((src, dst))
        return edge_set

    def set_dtype(self, dtype):
        self.dtype = dtype

    def has_edge(self, src, dst):
        return 0 if (src, dst) in self.edge_set else 1 # 假设self.labels = ['Yes', 'No']

    def __len__(self):
        return self.edge_index.size(1)

    def _process_batch(self, batch, node):
        """处理单个批次的辅助方法"""
        new_batch = batch.copy()
        new_batch.remove(node)
        new_batch = new_batch[:8]
        new_batch.insert(0, node)
        return new_batch

    def _get_subgraph_nodes(self, batch, subgraphs):
        """获取子图节点信息的辅助方法"""
        l = len(batch)
        inv = {batch[i]: i for i in range(l)}
        sg_nodes = torch.zeros(l, l)
        for i in range(l):
            for neigh in subgraphs[batch[i]]:
                if neigh in batch:
                    sg_nodes[inv[batch[i]], inv[neigh]] = 1
        return sg_nodes

    def _process_edges(self, edges, batch):
        """处理边信息的辅助方法"""
        inv = {batch[i]: i for i in range(len(batch))}
        edges = torch.tensor(edges)
        batch = torch.tensor(batch)
        mask_0 = torch.isin(edges[0], batch)
        mask_1 = torch.isin(edges[1], batch)
        mask = mask_0 & mask_1
        edges = edges[:, mask]
        # print(edges, inv)
        edges = [[inv[int(i)] for i in e] for e in edges]
        return torch.tensor(edges, dtype=torch.long)

    def __getitem__(self, idx):
        start = len(self.inst_head_ids)
        text_len = len(self.texts_ids[0])
        mid_len = len(self.inst_mid_ids)
        src = self.edge_index[0, idx].item()
        dst = self.edge_index[1, idx].item()

        # 处理批次
        batch1 = self._process_batch(self.batchs[self.valid_batchs[src]], src)
        batch2 = self._process_batch(self.batchs[self.valid_batchs[dst]], dst)

        # 对齐批次长度
        l1, l2 = len(batch1), len(batch2)
        # print(l1, l2)
        if l1 < l2:
            batch1.extend(batch2[l1:])
        elif l2 < l1:
            batch2.extend(batch1[l2:])

        # 生成输入 ID
        t_ids = torch.stack([
            torch.cat([
                self.inst_head_ids,
                self.texts_ids[batch1[i]],
                self.inst_mid_ids,
                self.texts_ids[batch2[i]],
                self.inst_tail_ids,
                self.label_ids[self.has_edge(batch1[i], batch2[i])]
            ], dim=0)
            for i in range(len(batch1))
        ])

        # 生成节点信息掩码
        node_info_mask = torch.zeros(len(t_ids[0]), dtype=torch.bool)
        node_info_mask[start : start + text_len * 2 + mid_len] = True
        node_info_mask[start + text_len : start + text_len + mid_len] = False

        # 生成注意力掩码
        attention_mask = torch.ones((t_ids.size(0), t_ids.size(1)))
        attention_mask[t_ids == self.tokenizer.pad_token_id] = 0

        # 获取子图节点信息
        sg_nodes1 = self._get_subgraph_nodes(batch1, self.subgraphs)
        sg_nodes2 = self._get_subgraph_nodes(batch2, self.subgraphs)
        sg_nodes = [sg_nodes1.tolist(), sg_nodes2.tolist()]
        sg_nodes = torch.tensor(sg_nodes)

        # 获取结构编码
        struct_encode = None if self.struct_encodes is None else self.struct_encodes[self.batchs[idx]]

        # 处理边信息
        edges1 = self._process_edges(self.edges[self.valid_batchs[src]], batch1)
        edges2 = self._process_edges(self.edges[self.valid_batchs[dst]], batch2)

        # 生成图数据
        rand = np.random.normal(loc=0, scale=1.0, size=(len(batch1), 20))
        x = torch.from_numpy(rand.astype('float32'))
        x[-1] = 0
        # graph1 = Data(x=x, edge_index=edges1)
        # graph2 = Data(x=x, edge_index=edges2)
        # graph = [graph1, graph2]

        return {
            "input_ids": t_ids,
            "attention_mask": attention_mask,
            "struct_encode": struct_encode,
            "subgraph_nodes": sg_nodes,
            "x": x,
            "edge_index": edges1,
            "edge_index2": edges2,
            "valid_nodes_mask": None,
            "labels": t_ids,
            "node_info_mask": node_info_mask,
            "mode": 1
        }


if __name__ == "__main__":
    earxiv = torch.load('./arxiv_edge_sdg.pt')
    # eproducts = torch.load('./products_edge_sdg.pt')
    arxiv = torch.load('./arxiv_sdg_dataset.pt')
    # products = torch.load('./products_sdg_dataset.pt')
    # cora = torch.load('./cora_sdg_dataset.pt')
    # pubmed = torch.load('./pubmed_sdg_dataset.pt')
    # wikics = torch.load('./wikics_sdg_dataset.pt')
    # ecora = torch.load('./cora_edge_sdg.pt')
    # epubmed = torch.load('./pubmed_edge_sdg.pt')
    # ewikics = torch.load('./wikics_edge_sdg.pt')
    # ewikics._append_edges()

    fs = MergedSDGDataset()
    fs.few_shot_init(arxiv, 10)
    n = len(fs)
    dtss = [fs, earxiv]# [earxiv, eproducts, ecora] # , epubmed, ewikics]
    nums = [n, n] #[200, 200, 100] #[400, 400] # , 80, 80, 80]
    merged = MergedSDGDataset()
    # m2 = MergedSDGDataset()
    # nmerged = MergedSDGDataset()
    merged.merge_init(dtss, nums)
    torch.save(merged, f'./arxiv_merged_sdg.pt')

    exit()

    dtss = [arxiv, products, cora] # , pubmed, wikics]
    nmerged.merge_init(dtss, nums)

    m2.merge_init([merged, nmerged], [500, 500])

    print(len(m2))
    print(m2[-1])
    torch.save(m2, f'./merged_fs_edge_sdg_acp.pt')

    # cora = torch.load('./cora_sdg_dataset.pt')
    # pubmed = torch.load('./pubmed_sdg_dataset.pt')
    # wikics = torch.load('./wikics_sdg_dataset.pt')
    # merged =  MergedSDGDataset(arxiv, products, 400, 400)
    # print(len(merged))
    # merged =  MergedSDGDataset(merged, cora, 800, 100)
    # print(len(merged))
    # merged =  MergedSDGDataset(merged, wikics, 900, 100)
    # print(len(merged))
    # torch.save(merged, './ap_sdg_dataset2.pt')
    # merged = MergedSDGDataset()
    # merged.few_shot_init(arxiv, 10)
    # print(merged[0], len(merged))
    # torch.save(merged, f'./pubmed_fs_sdg_dataset.pt')