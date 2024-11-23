from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer

class SDGDataset(Dataset):
    def __init__(self, texts, labels, struct_encodes, batchs, subgraphs, valid_nodes_masks, tokenizer, inst, split):
        self.struct_encodes = struct_encodes
        self.subgraphs = subgraphs
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.batchs = batchs
        self.valid_nodes_masks = valid_nodes_masks
        self.split = split

        self.label_ids = self.tokenizer(
            labels,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][:, 1:]

        self.texts_ids = self.tokenizer(
            texts,
            max_length=300,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )["input_ids"][1:]
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
        

    def __len__(self):
        return len(self.batchs)

    def __getitem__(self, idx):
        t_ids = torch.stack([
            torch.cat([self.inst_head_ids, self.texts_ids[i], self.inst_tail_ids, self.label_ids[i]], dim=0)
            for i in self.batchs[idx]
        ])
        attention_mask = torch.ones((t_ids.size(0), t_ids.size(1)))
        attention_mask[t_ids==self.tokenizer.pad_token_id] = 0
        return {
            "input_ids": t_ids,
            "attention_mask": attention_mask,
            "struct_encode": self.struct_encodes[idx],
            "subgraph_nodes": self.subgraphs[idx],
            "valid_nodes_mask": self.valid_nodes_masks[idx]
        }
