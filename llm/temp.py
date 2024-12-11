from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
import torch

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')

model = LlamaForCausalLM.from_pretrained('./Llama-2-7b-chat-hf')

model.to(device)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora

dataset = torch.load('/home/wangjingchu/code/SDGLM/cora_sdg_dataset.pt')
total_loss = 0
for i in range(140):
    data = dataset[i]
    with torch.no_grad():
        output = model(input_ids=data['input_ids'].to(device), attention_mask=data['attention_mask'].to(device), labels=data['labels'].to(device))
    loss = float(output['loss'])
    print(loss, loss/301)
    total_loss += loss

print(f'avg_loss: {total_loss/140:.6f}')