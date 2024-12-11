from gpse import GPSE
from gpse_mlp import GPSE_MLP
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
import sdg_dataset

dataset = torch.load(f'/home/wangjingchu/code/SDGLM/cora_sdg_dataset.pt')
model = torch.load(f'/home/wangjingchu/code/SDGLM/structure_encoder/cora_tag_pt_module.pt')

# sd = torch.load(f'/home/wangjingchu/code/SDGLM/llm/ckpt_tuning_gpse2/checkpoint-42120/pytorch_model.bin')
# new_state_dict = {}

# for key, value in sd.items():
#     if "gpsemlp." in key:
#         new_key = key.replace("gpsemlp.", "")
#         new_state_dict[new_key] = value

# model.load_state_dict(new_state_dict)
model.to('cpu')

total_loss = 0
max_loss = 0
min_loss = 1000
for data in dataset:
    graph = data['graph']
    with torch.no_grad():
        output = model(graph)
        loss = model.constractive_loss(graph, output)
    max_loss = max(loss, max_loss)
    min_loss = min(loss, min_loss)
    total_loss += loss
print(f'avg_loss:{total_loss/len(dataset):6f}')
print(f'max_loss:{max_loss:6f}')
print(f'min_loss:{min_loss:6f}')
