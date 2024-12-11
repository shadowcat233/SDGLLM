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

sdict = torch.load(f'/home/wangjingchu/code/SDGLM/llm/{arg.dir}/checkpoint-{arg.n}/pytorch_model.bin')
for d in sdict:
    if 'struct_proj' in  d:
        print(sdict[d])


import json
with open(f'/home/wangjingchu/code/SDGLM/llm/{arg.dir}/checkpoint-{arg.n}/trainer_state.json', 'r') as f:
    trainer_config = json.load(f)

history = trainer_config['log_history']
loss = torch.tensor([float(h['loss']) for h in history])
epoch = torch.tensor([float(h['epoch']) for h in history])

loss_per_epoch = []
loss_list = []
for i in range(len(loss)):
    if i==len(loss)-1 or (epoch[i] == int(epoch[i])+0.01 and epoch[i] != 0.01):
        if epoch[i] != 0:
            loss_mean = torch.tensor(loss_list).mean()
            loss_per_epoch.append(float(loss_mean))
        loss_list = []
    loss_list.append(loss[i])
print(loss_per_epoch)

# k=105
# num_batches = len(loss) // k
# if len(loss) % k != 0:
#     loss = loss[:num_batches * k]

# loss = loss.reshape((k, -1))

# loss = loss.mean(0)


import matplotlib.pyplot as plt
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
