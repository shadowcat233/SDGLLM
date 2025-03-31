
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from sdgllama_modeling4 import SDGLlamaForCausalLM, SDGConfig
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')

tokenizer = AutoTokenizer.from_pretrained('./Llama-2-7b-chat-hf')

sdg_config = SDGConfig(
    se_dim_in=256,
    proj_path=None,
    gpsemlp_path='../models_and_data/cora_gpse_256.pt',
    semantic_path=None,
    # has_gpsemlp=False, 
    # has_struct_proj=False,
    # has_semantic_proj=False,
    **pretrained_config.to_dict()
)

model = SDGLlamaForCausalLM.from_pretrained(
        './Llama-2-7b-chat-hf',
        config=sdg_config,
        torch_dtype=torch.float
).to(device)

path = './ckpt_tuning_gpse50/checkpoint-420/pytorch_model.bin'
new_state_dict = torch.load(path)

model_state_dict = model.state_dict()
model_state_dict.update(new_state_dict)
model.load_state_dict(model_state_dict)

# torch.save(model.model.struct_projector, '../models_and_data/struct_proj_n2_c_all.pt')
# torch.save(model.model.semantic_projector, '../models_and_data/semantic_proj_c_only.pt')
# torch.save(model.gpsemlp, '../models_and_data/gpsemlp_c_all.pt')

# model = LlamaForCausalLM.from_pretrained('./Llama-2-7b-chat-hf')

model.to(device)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
import sdg_dataset

from tqdm import tqdm

print('START TESTING!')

all_acc = []

dts = ['cora'] #['wikics', 'pubmed', 'cora', 'arxiv', 'products']
for name in dts:
    print('---------------------------------------')
    print(f'{name} node')
    dataset = torch.load(f'../{name}_sdg_dataset.pt')

    true_labels = []
    predict_labels = []
    total = 0
    correct = 0
    lab_len = len(dataset.label_ids[0])
    inp_len = dataset[0]['input_ids'].size(1)

    progress_bar = tqdm(range(500, 600), desc="testing_node", unit="data")

    for idx in progress_bar:
        data = dataset[idx]
        inp_len = data['input_ids'].size(1) - lab_len
        label = data['input_ids'][:,-lab_len:]
        valid_node_mask = data['valid_nodes_mask']
        label = label[valid_node_mask==1]
        with torch.no_grad():
            generate_ids = model.generate(input_ids=data['input_ids'][:, :-lab_len].to(device),
                                            subgraph_nodes=data['subgraph_nodes'].to(device),
                                            valid_nodes_mask=data['valid_nodes_mask'].to(device),
                                            attention_mask=data['attention_mask'][:, :-lab_len].to(device),
                                            node_info_mask=data['node_info_mask'][:-lab_len].to(device),
                                            x=data['x'].to(device),
                                            edge_index=data['edge_index'].to(device),
                                            # edge_index2=data['edge_index2'].to(device),
                                            mode=0,
                                            max_length=inp_len+10)[:,inp_len:]
            # loss = model(input_ids=data['input_ids'].to(device),
            #                                 subgraph_nodes=data['subgraph_nodes'].to(device),
            #                                 valid_nodes_mask=data['valid_nodes_mask'].to(device),
            #                                 attention_mask=data['attention_mask'].to(device),
            #                                 node_info_mask=data['node_info_mask'].to(device),
            #                                 graph=data['graph'].to(device),
            #                                 labels=data['labels'].to(device),
            #                                 use_gpsemlp=True,
            #                                 use_struct_projector=True,
            #                                 use_semantic_proj=True)['loss']
            # print(loss.item())
        valid_ids = generate_ids[valid_node_mask==1]

        for i in range(label.size(0)):
            token_ids = label[i].tolist()
            ground_truth = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            true_labels.append(ground_truth)

            res = tokenizer.decode(valid_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            res = res.strip()
            res = res.translate(str.maketrans("", "", ".\'\""))
            predict_labels.append(res)

            total += 1
            if ground_truth in res: correct += 1
            progress_bar.set_postfix({"ACC": correct/total})
            # print(f'{idx:4d}: predict label: {res} | ground truth: {ground_truth} ACC: {(correct/total):.4f}')

    accuracy = sum(1 for a, b in zip(true_labels, predict_labels) if a == b) / len(true_labels)
    acc = correct/total
    all_acc.append(acc)
    print(f"Accuracy: {acc:.4f}    Strict Accuracy: {accuracy:.4f}")

# dts = ['pubmed', 'cora', 'wikics', 'arxiv', 'products']
for name in dts:
    print('---------------------------------------')
    print(f'{name} edge')
    dataset = torch.load(f'../{name}_edge_sdg.pt')
    dataset._append_edges()
    # print(len(dataset))

    progress_bar = tqdm(range(500, 600), desc="testing_edge", unit="data")

    true_labels = []
    predict_labels = []
    total = 0
    correct = 0
    lab_len = len(dataset.label_ids[0])
    # print(lab_len)
    inp_len = dataset[0]['input_ids'].size(1)

    # for idx in tqdm(range(200, 300), desc="testing", unit="data"):
    for idx in progress_bar:
        data = dataset[idx]
        inp_len = data['input_ids'].size(1) - lab_len
        label = data['input_ids'][0,-lab_len:]
        with torch.no_grad():
            generate_ids = model.generate(input_ids=data['input_ids'][:, :-lab_len].to(device),
                                            subgraph_nodes=data['subgraph_nodes'].to(device),
                                            # valid_nodes_mask=data['valid_nodes_mask'].to(device),
                                            attention_mask=data['attention_mask'][:, :-lab_len].to(device),
                                            node_info_mask=data['node_info_mask'][:-lab_len].to(device),
                                            x=data['x'].to(device),
                                            edge_index=data['edge_index'].to(device),
                                            edge_index2=data['edge_index2'].to(device),
                                            mode=1,
                                            max_length=inp_len+10)[:,inp_len:]
        valid_ids = generate_ids[0]

        i = 0
        token_ids = label[i].tolist()
        ground_truth = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        true_labels.append(ground_truth)

        res = tokenizer.decode(valid_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        res = res.strip()
        res = res.translate(str.maketrans("", "", ".\'\""))
        predict_labels.append(res)

        total += 1
        if ground_truth in res: correct += 1
        progress_bar.set_postfix({"ACC": correct/total})
        # print(f'{idx:4d}: predict label: {res} | ground truth: {ground_truth} ACC: {(correct/total):.4f}')

    accuracy = sum(1 for a, b in zip(true_labels, predict_labels) if a == b) / len(true_labels)
    acc = correct/total
    all_acc.append(acc)
    print(f"Accuracy: {(correct/total):.4f}    Strict Accuracy: {accuracy:.4f}")

print(path)
print(all_acc)

# import pandas as pd
# df = pd.DataFrame({'predict labels': predict_labels, 'true labels': true_labels})
# df.to_csv('test_tuning_gpse2_42120_all.csv', index=False)


# # print('====================================')
# # for name, param in model.named_parameters():
# #     if "weight" in name and ("31" in name or "struct" in name):
# #         print(name, param.shape)

# # estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=5, num_nodes=1)