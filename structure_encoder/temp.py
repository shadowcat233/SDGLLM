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

model2 = model

sd = torch.load(f'/home/wangjingchu/code/SDGLM/llm/ckpt_tuning_gpse2/checkpoint-42120/pytorch_model.bin')
g_sd = {}

for key, value in sd.items():
    if "gpsemlp." in key:
        new_key = key.replace("gpsemlp.", "")
        g_sd[new_key] = value

model2.load_state_dict(g_sd)
model.to('cpu')
model2.to('cpu')

import torch.nn.functional as F
def sim(z):
    z = F.normalize(z) 
    return torch.mm(z, z.t()) 

# total_loss = 0
# max_loss = 0
# min_loss = 1000

from torch.nn import MSELoss
loss_fct = MSELoss()

total_loss1 = total_loss2 = 0
for i in range(len(dataset)):
    data = dataset[i]
    graph = data['graph']
    with torch.no_grad():
        output1 = model(graph)
        output2 = model2(graph)
    num_nodes = len(graph.x)
    A = torch.zeros((num_nodes, num_nodes))  
    A[graph.edge_index[0], graph.edge_index[1]] = 1  
    A[graph.edge_index[1], graph.edge_index[0]] = 1

    loss1 = loss_fct(sim(output1), A)
    loss2 = loss_fct(sim(output2), A)

    total_loss1 += loss1.item()
    total_loss2 += loss2.item()

    if (1+i)%10 == 0:
        print(total_loss1/(i+1), total_loss2/(i+1))

    #     loss = model.constractive_loss(graph, output)
    # max_loss = max(loss, max_loss)
    # min_loss = min(loss, min_loss)
    # total_loss += loss

print(f'avg_loss1:{total_loss1/len(dataset):6f}')
print(f'avg_loss2:{total_loss2/len(dataset):6f}')
# print(f'max_loss:{max_loss:6f}')
# print(f'min_loss:{min_loss:6f}')
