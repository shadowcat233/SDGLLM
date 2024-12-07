import torch

# sdict = torch.load('/home/wangjingchu/code/SDGLM/llm/checkpoints/train3/epoch0/struct_proj.bin')

# print(sdict)

proj = torch.nn.Linear(1024, 4096, bias=False)
torch.nn.init.normal_(proj.weight, mean=0.0, std=1e-6)
print(proj.weight)
torch.save(proj, f'/home/wangjingchu/code/SDGLM/llm/checkpoints/struct_proj.pt')

# sdict = torch.load('/home/wangjingchu/code/SDGLM/llm/checkpoints_lr=0.01/checkpoint-27489/pytorch_model.bin')
# print(sdict)

# import json
# with open('/home/wangjingchu/code/SDGLM/llm/checkpoints_lr=0.01/checkpoint-27489/trainer_state.json', 'r') as f:
#     trainer_config = json.load(f)

# history = trainer_config['log_history']
# loss = torch.tensor([float(h['loss']) for h in history])

# k=70
# num_batches = len(loss) // k
# if len(loss) % k != 0:
#     loss = loss[:num_batches * k]

# loss = loss.reshape((k, -1))

# loss = loss.mean(0)


# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.plot(loss, marker='o', linestyle='-', color='b', label='loss')

# # 添加标题和标签
# plt.title('training loss')
# plt.xlabel('training epochs')
# plt.ylabel('loss')

# # 显示图例
# plt.legend()

# # 显示网格
# plt.grid(True)

# output_path = 'line_plot.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存为高分辨率图像
# print(f"图像已保存到: {output_path}")

# # 显示图像
# plt.show()
