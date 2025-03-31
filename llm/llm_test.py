# import os

# from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig, LlamaModel
# import torch
# from torch.nn import DataParallel

# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# upper_dir = os.path.dirname(current_dir)
# sys.path.append(upper_dir)
# from TAGLAS.datasets import Cora

# dataset = torch.load('../TAGDataset/cora/cora_tag.pt')
# data = dataset[0]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = 'cpu'
# model_path = './Llama-2-7b-chat-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(model_path).to(device)

# for name, param in model.named_parameters():
#     if "weight" in name:
#         print(name, param.shape)
# exit()
# # model = DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])  
# # model = model.to("cuda")

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # model.to(device)

# prompt_head = "[SYSTEM MESSAGE] A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, \
# and polite answers to the user's questions.\n\
# [USER] Here's the title and abstract of a paper, please tell me which category the paper belongs to."
# prompt_tail = "Optional Categories: Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods \
# \nPlease select one of the options from the above list that is the most likely category. Only answer the name of the category and don't add any other replies. \
# \n[ASSISTENT] The paper belongs to the category of "

# true_labels = [data.label[data.label_map[i]] for i in range(2708)]

# predicted_labels = []
 
# def print_cuda_info():
#     print(f"已用显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#     print(f"保留显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
#     print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

# for i in range(2708):
#     text = data.x[i]
#     prompt = prompt_head + '\n' + text + '\n' + prompt_tail
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     with torch.no_grad():
#         generate_ids = model.generate(inputs.input_ids, max_length=2000)
#     res = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

#     if res.startswith(prompt): res = res[len(prompt):]
#     res = res.translate(str.maketrans("", "", ".\'\""))
#     predicted_labels.append(res.strip())
    
#     print(f'{i:4d}:  predict label: {res.strip()}  |  true label: {data.label[data.label_map[i]]}')
#     # print_cuda_info()
#     # torch.cuda.empty_cache()

 
 
# accuracy = sum(1 for a, b in zip(true_labels, predicted_labels) if a == b) / len(true_labels)
# print(f"Accuracy: {accuracy:.4f}")

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# model_path = './Llama-2-7b-chat-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# config = LlamaConfig.from_pretrained(model_path)
# decoder = LlamaDecoderLayer(config, 0)


from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from sdgllama_modeling4 import SDGLlamaForCausalLM, SDGConfig
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

tokenizer = AutoTokenizer.from_pretrained('./Llama-2-7b-chat-hf')

model = LlamaForCausalLM.from_pretrained('./Llama-2-7b-chat-hf').to(device)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
import sdg_dataset

from tqdm import tqdm

print('START TESTING!')

# dts = ['pubmed', 'cora', 'wikics', 'arxiv', 'products']
# for name in dts:
#     print('---------------------------------------')
#     print(name)
#     dataset = torch.load(f'../{name}_sdg_dataset.pt')

#     true_labels = []
#     predict_labels = []
#     total = 0
#     correct = 0
#     lab_len = len(dataset.label_ids[0])
#     inp_len = dataset[0]['input_ids'].size(1)
#     for idx in tqdm(range(200, 400), desc="testing", unit="data"):
#         data = dataset[idx]
#         inp_len = data['input_ids'].size(1) - lab_len
#         label = data['input_ids'][:,-lab_len:]
#         valid_node_mask = data['valid_nodes_mask']
#         input_ids =data['input_ids'][valid_node_mask==1]
#         label = label[valid_node_mask==1]
#         with torch.no_grad():
#             generate_ids = model.generate(input_ids=input_ids[:, :-lab_len].to(device),
#                                             attention_mask=data['attention_mask'][valid_node_mask==1][:, :-lab_len].to(device),
#                                             max_length=inp_len+20)[:,inp_len:]

#         for i in range(label.size(0)):
#             token_ids = label[i].tolist()
#             ground_truth = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
#             true_labels.append(ground_truth)

#             res = tokenizer.decode(generate_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
#             res = res.strip()
#             res = res.translate(str.maketrans("", "", ".\'\""))
#             predict_labels.append(res)

#             total += 1
#             if ground_truth in res: correct += 1
#             # print(f'{idx:4d}: predict label: {res} | ground truth: {ground_truth} ACC: {(correct/total):.4f}')

#     accuracy = sum(1 for a, b in zip(true_labels, predict_labels) if a == b) / len(true_labels)
#     print(f"Accuracy: {(correct/total):.4f}    Strict Accuracy: {accuracy:.4f}")


dts = ['wikics']
for name in dts:
    print('---------------------------------------')
    print(name)
    dataset = torch.load(f'../{name}_edge_sdg.pt')
    dataset._append_edges()
    print(len(dataset))

    true_labels = []
    predict_labels = []
    total = 0
    correct = 0
    lab_len = len(dataset.label_ids[0])
    print(lab_len)
    inp_len = dataset[0]['input_ids'].size(1)

    # for idx in tqdm(range(200, 300), desc="testing", unit="data"):
    for idx in range(500, 600):
        data = dataset[idx]
        inp_len = data['input_ids'].size(1) - lab_len
        label = data['input_ids'][0,-lab_len:]
        input_ids=data['input_ids'][0, :-lab_len].unsqueeze(0).to(device)
        with torch.no_grad():
            generate_ids = model.generate(input_ids=input_ids,
                                            attention_mask=data['attention_mask'][0, :-lab_len].unsqueeze(0).to(device),
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
        print(f'{idx:4d}: predict label: {res} | ground truth: {ground_truth} ACC: {(correct/total):.4f}')

    accuracy = sum(1 for a, b in zip(true_labels, predict_labels) if a == b) / len(true_labels)
    print(f"Accuracy: {(correct/total):.4f}    Strict Accuracy: {accuracy:.4f}")