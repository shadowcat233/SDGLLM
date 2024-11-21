import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora


class SDG_LLM(nn.Module):
    def __init__(self, llama_model_name, num_layers=3, p_dim_in=128, p_dim_out=768):
        super(SDG_LLM, self).__init__()
        self.llm = LlamaForCausalLM.from_pretrained(llama_model_name)
        self.projector = nn.Linear(p_dim_in, self.llm.config.hidden_size)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_embeddings = self.llm.get_input_embeddings()
        self.decoder = self.llm.get_decoder()
        self.num_layers = num_layers

        for param in self.llm.parameters():
            param.requires_grad = False

    def get_special_embeddings(self, text):
        tokens = self.tokenizer(text, return_tensors="pt")['input_ids']
        return self.input_embeddings(tokens.to(self.projector.weight.device))

    def sim(self, z1, z2):
        z1 = F.normalize(z1) 
        z2 = F.normalize(z2) 
        return torch.mm(z1, z2.t()) 

    def structure_attention(self, t, sim_m):
        sim_m_expanded = sim_m.unsqueeze(-1).unsqueeze(-1) 
        weighted_t = (sim_m_expanded * t.unsqueeze(0)).sum(dim=1)
        sim_sum = sim_m.sum(dim=1, keepdim=True).unsqueeze(-1)
        sim_sum = sim_sum + (sim_sum == 0).float()
        t_hat = weighted_t / sim_sum
        return t_hat

    def forward(self, t, p, inst):
        p_hat = self.projector(p)  # 将结构化信息映射到 LLaMA 兼容的维度

        sim_m = self.sim(p_hat, p_hat)  # 计算相似度矩阵
        t_prime = self.decoder(input_ids=t).last_hidden_state  # 提取文本表示
        for _ in range(self.num_layers):
            t_prime = self.structure_attention(t_prime, sim_m)
            t_prime = self.decoder(inputs_embeds=t_prime).last_hidden_state

        head_embeds = self.get_special_embeddings(inst['head'])
        tail_embeds = self.get_special_embeddings(inst['tail'])
        target_node_textual_info_embeds = self.get_special_embeddings("target node textual information: ")
        target_node_structural_info_embeds = self.get_special_embeddings("\ntarget node structual information: ")
        textual_info_embeds = self.get_special_embeddings("\nother node textual information: ")
        structural_info_embeds = self.get_special_embeddings("\nother node structual information: ")

        t_prime_mean_first = t_prime.mean(dim=1)[0].unsqueeze(0).unsqueeze(0) 
        t_prime_mean_rest = t_prime.mean(dim=1)[1:].unsqueeze(1)

        p_hat_first = p_hat.unsqueeze(1)[0].unsqueeze(0) 
        p_hat_rest = p_hat.unsqueeze(1)[1:] 

        combined_embeds = torch.cat([
            head_embeds,
            target_node_textual_info_embeds,
            t_prime_mean_first,
            target_node_structural_info_embeds,
            p_hat_first,
            textual_info_embeds
        ], dim=1)

        # for i in range(t_prime_mean_rest.size(0)):
        #     combined_embeds = torch.cat([combined_embeds, t_prime_mean_rest[i].unsqueeze(0)], dim=1)
        # combined_embeds = torch.cat([combined_embeds, structural_info_embeds], dim=1)
        # for i in range(p_hat_rest.size(0)):
        #     combined_embeds = torch.cat([combined_embeds, p_hat_rest[i].unsqueeze(0)], dim=1)
        combined_embeds = torch.cat([combined_embeds, tail_embeds], dim=1)

        # combined_embeds = torch.cat([
        #     head_embeds,
        #     target_node_textual_info_embeds,
        #     t_prime_mean_first,
        #     target_node_structural_info_embeds,
        #     p_hat_first,
        #     textual_info_embeds,
        #     t_prime_mean_rest,
        #     structural_info_embeds,
        #     p_hat_rest,
        #     tail_embeds
        # ], dim=1)

        # logits = self.llm(inputs_embeds=combined_embeds, attention_mask=attention_mask).logits

        # return logits
        return combined_embeds

    def logits_to_text(self, logits):
        tokenizer = self.tokenizer
        if len(logits.size()) == 2:
            logits = logits.unsqueeze(0)
        decoded_texts = []
        for i in range(logits.size(0)):
            sample_logit = logits[i]
            # sample_mask = masks[i]
            # sample_logit = sample_logit[sample_mask]
            token_ids = sample_logit[:, :32000].argmax(dim=-1).unsqueeze(0)
            token_ids[token_ids >= 32000] = 1
            sample_text = tokenizer.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            decoded_texts.extend(sample_text)
        return decoded_texts

    def embeddings_generate(self, embeddings):
        max_generate_steps = 50  # 最多生成 50 个 token
        generated_tokens = []
        current_embeds = embeddings

        for step in range(max_generate_steps):
            outputs = self.llm(inputs_embeds=current_embeds)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1) 
            print(next_token_id)
            generated_tokens.append(next_token_id.item())

            next_token_embed = self.input_embeddings(next_token_id.unsqueeze(-1))  # (batch_size, 1, hidden_dim)
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)

            # attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
        decoded_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded_text

    def generate(self, raw_texts, p, inst, device):
        t = self.tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True)['input_ids'].to(device)
        with torch.no_grad():
            embeddings = self(t, p, inst)
            decoded_text = self.embeddings_generate(embeddings)
        return decoded_text

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu") 
sdg = SDG_LLM('/home/wangjingchu/code/SDGLM/llm/Llama-2-7b-chat-hf').to(device)
subgraph_nodes = torch.load('/home/wangjingchu/code/SDGLM/TAGDataset/cora/subgraph_nodes.pt')
se_output = torch.load('/home/wangjingchu/code/SDGLM/structure_encoder/output_cora_tag_pt_module.pt').to(device)
cora_dataset = torch.load('/home/wangjingchu/code/SDGLM/TAGDataset/cora/cora_tag.pt')
data = cora_dataset[0]

raw_texts = [str(data.x[i]) for i in subgraph_nodes[0]]
p = torch.stack([se_output[i] for i in subgraph_nodes[0]])


inst = {
    'head': "Graph description: Each node represents a paper, edges represent citations. Below are the textual and structural information of target node and the other nodes within the subgraph of the target node.\n",
    'tail': "Categories: Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods.\n Please choose the category of the target node from the category list. Please only answer the category's name.\nAnswer: "
}

logits = sdg.generate(raw_texts, p, inst, device)
print(logits)


