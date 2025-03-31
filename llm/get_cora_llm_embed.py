from transformers import LlamaForCausalLM, AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from sdgllama_modeling4 import SDGLlamaForCausalLM, SDGConfig
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

tokenizer = AutoTokenizer.from_pretrained('./Llama-2-7b-chat-hf')

pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')

sdg_config = SDGConfig(
    se_dim_in=256,
    proj_path=None,
    gpsemlp_path=None,
    semantic_path=None,
    has_gpsemlp=False, 
    has_struct_proj=False,
    has_semantic_proj=False,
    **pretrained_config.to_dict()
)

model = SDGLlamaForCausalLM.from_pretrained(
        './Llama-2-7b-chat-hf',
        config=sdg_config,
        torch_dtype=torch.float
).to(device)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora
cora = torch.load('../products_tag.pt')
data = cora[0]

# categories = [lab for lab in data.label if lab not in {'Yes', 'No', 'MISSING'}]
# categories_str = ', '.join(map(str, categories))
# inst = {}
# inst['head'] = f"Here's the title and abstract of a paper, please tell me which category it belongs to.\n"
# inst['tail'] = f"\nOptional Categories: {categories_str}"

model_path = './Llama-2-7b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.unk_token

def _tokenize(texts, max_length=None):
    """分词处理的辅助方法"""
    return tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )["input_ids"] 

texts_ids = _tokenize(data.x, 120)[:, :]
# inst_head_ids = _tokenize(inst['head']).squeeze(0)[0:]
# inst_tail_ids = _tokenize(inst['tail']).squeeze(0)[1:]

# input_ids = torch.stack([
#             torch.cat([inst_head_ids, texts_ids[i], inst_tail_ids], dim=0)
#             for i in range(len(data.x))
#         ])
input_ids = texts_ids
print(input_ids.shape)

def tokens2str(tensor):
    tensor_list = tensor.tolist()
    str_list = [str(i) for i in tensor_list]
    return '-'.join(str_list)

from tqdm import tqdm
from sklearn.decomposition import PCA

pca = PCA(n_components=512)

dts = ['products']
lab_dict = {}
x = []
y = []
temp = []
for name in dts:
    print('---------------------------------------')
    print(name)

    for idx in tqdm(range(len(input_ids)), desc="proceeding", unit="data"):
        with torch.no_grad():
            hidden_state = model(input_ids=input_ids[idx].unsqueeze(0).to(device),use_gpsemlp=False)['hidden_states']

        hidden_state = torch.flatten(hidden_state, start_dim=1)
        for h in hidden_state:
            temp.append(h.cpu().numpy())

    temp = np.array(temp)
    embed = pca.fit_transform(temp)
    x.extend(embed.tolist())
    print(len(x))
    print(x[0])
        

torch.save(torch.tensor(x), '../products_llm_x.pt')