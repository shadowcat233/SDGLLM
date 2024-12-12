
# from transformers import LlamaForCausalLM, AutoTokenizer
# from transformers.models.llama.configuration_llama import LlamaConfig
# from sdgllama_modeling2 import SDGLlamaForCausalLM, SDGConfig
# import torch

# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
# # device = 'cpu'

# pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')

# tokenizer = AutoTokenizer.from_pretrained('./Llama-2-7b-chat-hf')

# sdg_config = SDGConfig(
#     se_dim_in=1024,
#     proj_path=None,
#     gpsemlp_path='/home/wangjingchu/code/SDGLM/structure_encoder/cora_tag_pt_module.pt',
#     **pretrained_config.to_dict()
# )

# model = SDGLlamaForCausalLM.from_pretrained(
#         './Llama-2-7b-chat-hf',
#         config=sdg_config,
#         torch_dtype=torch.float
# )

# new_state_dict = torch.load('/home/wangjingchu/code/SDGLM/llm/ckpt_tuning_gpse2/checkpoint-42120/pytorch_model.bin')

# model_state_dict = model.state_dict()
# model_state_dict.update(new_state_dict)
# model.load_state_dict(model_state_dict)

# model.to(device)

# import os
# import sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# upper_dir = os.path.dirname(current_dir)
# sys.path.append(upper_dir)
# from TAGLAS.datasets import Cora

# dataset = torch.load('/home/wangjingchu/code/SDGLM/cora_sdg_dataset.pt')
# total_loss = 0
# total_g_loss = 0
# for idx in range(1):
#     data = dataset[idx]
#     valid_node_mask = data['valid_nodes_mask']
#     with torch.no_grad():
#         output = model(input_ids=data['input_ids'].to(device),
#                         struct_encode=data['struct_encode'].to(device),
#                         subgraph_nodes=data['subgraph_nodes'].to(device),
#                         valid_nodes_mask=data['valid_nodes_mask'].to(device),
#                         attention_mask=data['attention_mask'].to(device),
#                         graph=data['graph'].to(device),
#                         labels=data['labels'].to(device),
#                         use_struct_projector = True,
#                         use_gpsemlp = True,
#                         return_all_hidden_states = True,
#                         output_hidden_states = True,)
#     torch.save(output, './checkpoints/output_use_gpse&sp.pt')
#     loss = (output['loss'])
#     print(loss)
    # total_loss += loss
    # total_g_loss += g_loss

# print(f'avg_loss: {total_loss/140:.6f}')
# print(f'avg_g_loss: {total_g_loss/140:.6f}')

import torch
import torch.nn.functional as F


output = torch.load(f'./checkpoints/output_use_sp.pt', map_location='cpu')
hidden_states = output['hidden_states']
loss = float(output['loss'])
print(f'loss: {loss:4f}')


# el = []
# cs = []

for i in range(len(hidden_states) - 1):

    hidden_states_before = hidden_states[i]
    hidden_states_after = hidden_states[i+1]

    euclidean_loss = torch.norm(hidden_states_before - hidden_states_after, p=2, dim=-1)  # [batch_size, seq_len]

    cosine_similarity = F.cosine_similarity(hidden_states_before, hidden_states_after, dim=-1)  # [batch_size, seq_len]


    mean_euclidean_loss = euclidean_loss.mean()
    mean_cosine_similarity = cosine_similarity.mean()

    # el.append(mean_euclidean_loss.item())
    # cs.append(mean_cosine_similarity.item())

    print(f"{i} to {i+1} Mean Euclidean Loss: {mean_euclidean_loss.item():.6f}, Mean Cosine Similarity: {mean_cosine_similarity.item():.6f}")
