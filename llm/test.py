
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from sdgllama_modeling2 import SDGLlamaForCausalLM, SDGConfig
import torch

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')

tokenizer = AutoTokenizer.from_pretrained('./Llama-2-7b-chat-hf')

sdg_config = SDGConfig(
    se_dim_in=1024,
    proj_path=None,
    gpsemlp_path='/home/wangjingchu/code/SDGLM/structure_encoder/cora_tag_pt_module.pt',
    **pretrained_config.to_dict()
)

model = SDGLlamaForCausalLM.from_pretrained(
        './Llama-2-7b-chat-hf',
        config=sdg_config,
        torch_dtype=torch.float
)

new_state_dict = torch.load('/home/wangjingchu/code/SDGLM/llm/ckpt_tuning_gpse/checkpoint-16860/pytorch_model.bin')

for name, module in model.named_modules(): 
    if name in new_state_dict:
        try:
            # 加载新的 state_dict 权重到当前模块
            print(f"Loading weights for module: {name}")
            module.load_state_dict(new_state_dict[name])
        except Exception as e:
            print(f"Failed to load weights for {name}: {e}")

model.to(device)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora

dataset = torch.load('/home/wangjingchu/code/SDGLM/cora_sdg_dataset.pt')
print('START TESTING!')

true_labels = []
predict_labels = []
total = 0
correct = 0
for idx in range(2103, 2575):
    data = dataset[idx]
    inp_len = data['input_ids'].size(1) - 6
    label = data['input_ids'][:,-6:]
    valid_node_mask = data['valid_nodes_mask']
    label = label[valid_node_mask==1]
    # print(valid_node_mask)
    # print(label)
    generate_ids = model.generate(input_ids=data['input_ids'][:, :-6].to(device),
                                    struct_encode=data['struct_encode'].to(device),
                                    subgraph_nodes=data['subgraph_nodes'].to(device),
                                    valid_nodes_mask=data['valid_nodes_mask'].to(device),
                                    attention_mask=data['attention_mask'][:, :-6].to(device),
                                    graph=data['graph'].to(device),
                                    max_length=2000)[:,inp_len:]
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
        if res==ground_truth: correct += 1
        print(f'{idx:4d}: predict label: {res} | ground truth: {ground_truth} ACC: {(correct/total):.4f}')

accuracy = sum(1 for a, b in zip(true_labels, predicted_labels) if a == b) / len(true_labels)
print(f"Accuracy: {accuracy:.4f}")


# print('====================================')
# for name, param in model.named_parameters():
#     if "weight" in name and ("31" in name or "struct" in name):
#         print(name, param.shape)

# estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=5, num_nodes=1)