import torch

# sdict = torch.load('/home/wangjingchu/code/SDGLM/llm/checkpoints/train3/epoch0/struct_proj.bin')

# print(sdict)

# proj = torch.nn.Linear(1024, 4096, bias=False)
# torch.nn.init.normal_(proj.weight, mean=0.0, std=1e-6)
# print(proj.weight)
# torch.save(proj, f'/home/wangjingchu/code/SDGLM/llm/checkpoints/struct_proj.pt')

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="打印包含特定关键词的模型参数")
    parser.add_argument('--n', type=int, default=843)
    parser.add_argument('--dir', type=str, default='ckpt_tuning_gpse')
    return parser.parse_args()

arg = parse_args()

sdict = torch.load(f'./{arg.dir}/checkpoint-{arg.n}/pytorch_model.bin')
print(sdict.keys())
if 'model.sba_temp' in sdict:
    print(sdict['model.sba_temp'])
if 'model.struct_projector.weight' in sdict:
    weight = sdict['model.struct_projector.weight']
    print(weight, weight.shape)
if 'model.semantic_projector.weight' in sdict:
    weight = sdict['model.semantic_projector.weight']
    print(weight, weight.shape)
    # proj = torch.nn.Linear(256, 4096, bias=False)
    # proj.weight.data = weight
    # print(proj.state_dict())
    # torch.save(proj, './struct_proj_256.pt')

# for d in sdict:
#     if 'struct' or 'semantic' in d:
#         print(d)
# print(sdict['model.graph_token_embedding'])
# print(sdict['model.text_token_embedding'])


import json
with open(f'./{arg.dir}/checkpoint-{arg.n}/trainer_state.json', 'r') as f:
    trainer_config = json.load(f)

history = trainer_config['log_history']
loss = torch.tensor([float(h['loss']) for h in history])
epoch = torch.tensor([float(h['epoch']) for h in history])

loss_per_epoch = []
loss_list = []
last = len(loss)-1
e = 1
for i in range(len(loss)):
    if i==last or epoch[i] >= e:
        loss_mean = torch.tensor(loss_list).mean()
        loss_per_epoch.append(float(loss_mean))
        loss_list = []
        e += 1
    loss_list.append(loss[i])

print(loss_per_epoch)


k=1
num_batches = len(loss) // k
if len(loss) % k != 0:
    loss = loss[:num_batches * k]

loss_per_10k_steps = loss.reshape((-1, k))
loss_per_10k_steps = loss_per_10k_steps.mean(1)


import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(12, 6))
plt.plot(loss_per_epoch, marker='o', linestyle='-', color='b', label='loss')

plt.title(f'training loss of {arg.dir}')
plt.xlabel('training epochs')
plt.ylabel('loss')

plt.legend()

plt.grid(True)

output_path = f'loss_per_epoch_of_{arg.dir}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存到: {output_path}")

plt.show()

plt.figure(figsize=(8, 6))
plt.plot(np.arange(10*k, len(loss_per_10k_steps) * 10*k+10*k, 10*k), loss_per_10k_steps, marker='o', linestyle='-', color='b', label='loss')

plt.title(f'training loss')
plt.xlabel('training steps')
plt.ylabel('loss')

plt.legend()
plt.grid(True)

output_path = f'loss_per_10k_steps_of_{arg.dir}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存到: {output_path}")
