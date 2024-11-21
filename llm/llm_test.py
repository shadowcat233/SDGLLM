import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig
import torch
from torch.nn import DataParallel

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora

dataset = torch.load('../TAGDataset/cora/cora_tag.pt')
data = dataset[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path).to(device)

# model = DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])  
# model = model.to("cuda")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
 
prompt_head = "Here's the title and abstruct of a paper, please tell me which category the paper belongs to."
prompt_tail = "Optional Categories: Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods \
\nPlease select one of the options from the above list that is the most likely category. Only answer the name of the category and don't add any other replies. \
\nYour answer is: "

true_labels = [data.label[data.label_map[i]] for i in range(140)]

predicted_labels = []
 
def print_cuda_info():
    print(f"已用显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"保留显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

for i in range(140):
    text = data.x[i]
    prompt = prompt_head + '\n' + text + '\n' + prompt_tail
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generate_ids = model.generate(inputs.input_ids, max_length=2000)
    res = tokenizer.decode(generate_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    if res.startswith(prompt): res = res[len(prompt):]
    predicted_labels.append(res.strip())
    
    print(f'{i:4d}:  predict label: {res.strip()}  |  true label: {data.label[data.label_map[i]]}')
    print_cuda_info()
    torch.cuda.empty_cache()

 
 
accuracy = sum(1 for a, b in zip(true_labels, predicted_labels) if a == b) / len(true_labels)
print(f"Accuracy: {accuracy:.4f}")

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# model_path = './Llama-2-7b-chat-hf'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# config = LlamaConfig.from_pretrained(model_path)
# decoder = LlamaDecoderLayer(config, 0)