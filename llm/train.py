import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM
import torch.nn.functional as F

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)
from TAGLAS.datasets import Cora


from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class SubgraphDataset(Dataset):
    def __init__(self, subgraph_nodes, labels):
        self.subgraph_nodes = subgraph_nodes
        self.labels = labels

    def __len__(self):
        return len(self.subgraph_nodes)

    def __getitem__(self, idx):
        return {'subgraph_nodes': self.subgraph_nodes[idx], 'labels': self.labels[idx]}


from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    subgraph_nodes = [torch.tensor(item['subgraph_nodes']) for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 使用 padding 对 subgraph_nodes 进行填充
    padded_subgraph_nodes = pad_sequence(subgraph_nodes, batch_first=True, padding_value=-1)
    labels = torch.stack(labels)  # 拼接标签
    
    return {'subgraph_nodes': padded_subgraph_nodes, 'labels': labels}

class SDG_LLM(nn.Module):
    def __init__(self, llama_model_name, num_layers=1, p_dim_in=128):
        super(SDG_LLM, self).__init__()
        self.llm = LlamaForCausalLM.from_pretrained(llama_model_name)
        for param in self.llm.parameters():
            param.requires_grad = False

        self.projector = nn.Linear(p_dim_in, self.llm.config.hidden_size)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.input_embeddings = self.llm.get_input_embeddings()
        self.decoder = self.llm.get_decoder()
        self.num_layers = num_layers

        for param in self.projector.parameters():
            param.requires_grad = True

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

    def forward(self, t, p):

        p_hat = self.projector(p)
        torch.cuda.empty_cache()
        print_cuda_info()
        sim_m = self.sim(p_hat, p_hat)

        t_prime = self.decoder(input_ids=t).last_hidden_state
        torch.cuda.empty_cache()
        print_cuda_info()

        for _ in range(self.num_layers):
            t_prime = self.structure_attention(t_prime, sim_m)
            t_prime = self.decoder(inputs_embeds=t_prime).last_hidden_state
            torch.cuda.empty_cache()
            print_cuda_info()

        return t_prime, p_hat

    def combine_embeddings(self, t_prime, p_hat, inst, only_target=True):
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

        if only_target==False:
            for i in range(t_prime_mean_rest.size(0)):
                combined_embeds = torch.cat([combined_embeds, t_prime_mean_rest[i].unsqueeze(0)], dim=1)
            combined_embeds = torch.cat([combined_embeds, structural_info_embeds], dim=1)
            for i in range(p_hat_rest.size(0)):
                combined_embeds = torch.cat([combined_embeds, p_hat_rest[i].unsqueeze(0)], dim=1)

        combined_embeds = torch.cat([combined_embeds, tail_embeds], dim=1)
        return combined_embeds

    def generate_from_embeddings(self, embeddings):

        max_generate_steps = 50
        generated_tokens = []
        all_logits = []
        current_embeds = embeddings
        attention_mask = torch.ones((embeddings.size(0), embeddings.size(1)), device=embeddings.device)

        for step in range(max_generate_steps):
            outputs = self.llm(inputs_embeds=current_embeds, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :] 
            next_token_id = torch.argmax(logits, dim=-1)
            generated_tokens.append(next_token_id)
            all_logits.append(logits)

            next_token_embed = self.input_embeddings(next_token_id.unsqueeze(-1))
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1)

            if (next_token_id == self.tokenizer.eos_token_id).all():
                break

        generated_tokens = torch.stack(generated_tokens, dim=1)  # (batch_size, seq_len)
        all_logits = torch.stack(all_logits, dim=1)

        return generated_tokens, all_logits

    def compute_normalized_loss(self, all_logits, labels, batch_size, pad_token_id):
        """
        计算归一化的损失，结合 batch_size 和 seq_len 的归一化逻辑。
        """
        all_logits = [torch.tensor(logit, dtype=torch.float16) if isinstance(logit, (int, float)) else logit
                        for logit in all_logits]
        flat_logits = torch.cat(all_logits, dim=0).view(-1, all_logits[0].size(-1))
        flat_labels = labels.view(-1).to(flat_logits.device)

        if flat_logits.size(0) > flat_labels.size(0):
            padding_length = flat_logits.size(0) - flat_labels.size(0)
            flat_labels = F.pad(flat_labels, (0, padding_length), value=pad_token_id)
        elif flat_logits.size(0) < flat_labels.size(0):
            padding_length = flat_labels.size(0) - flat_logits.size(0)
            padding = torch.zeros((padding_length, flat_logits.size(-1)), device=flat_logits.device)
            flat_logits = torch.cat([flat_logits, padding], dim=0)
        
        seq_len = labels.size(1)

        # 计算有效的目标 Token 数量（忽略 padding）
        effective_tokens = flat_labels.ne(pad_token_id).sum().item()

        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='sum')

        # 计算原始 Loss
        original_loss = loss_fn(flat_logits, flat_labels)

        # 归一化 Loss
        normalized_loss = original_loss * effective_tokens / (batch_size * seq_len)

        return normalized_loss


    def batch_generate_and_compute_loss(self, texts, p_encode, subgraph_nodes, inst, labels=None):
        """
        输入批量数据进行生成并计算损失。
        texts: List[str], 每个节点的自然语言信息。
        p_encode: torch.Tensor,大小为 (num_nodes, p_dim_in), 节点的位置结构信息。
        subgraph_nodes: List[List[int]], 每个节点的子图中节点编号。
        labels: torch.Tensor (optional), 目标文本对应的 token 序列, 用于计算损失。
        """
        batch_size = len(subgraph_nodes)
        device = p_encode.device

        all_logits = []
        all_generated_texts = []

        t_ids = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)['input_ids']

        for i in range(batch_size):
            subgraph_idxs = subgraph_nodes[i][subgraph_nodes[i] != -1]  # 当前节点的子图索引
            t = t_ids[subgraph_idxs]  # 子图的文本 token IDs
            p = p_encode[subgraph_idxs]  # 子图的结构化信息

            print(f'subgraph_idxs: {subgraph_idxs}')
            print(f't: {t}')
            print(f'p: {p}')

            # Forward pass
            t_prime, p_hat = self.forward(t, p)

            print(f't_p: {t_prime}')
            print(f'p_h: {p_hat}')

            # Combine embeddings for generation
            combined_embeds = self.combine_embeddings(t_prime, p_hat, inst)

            # Generate text and compute logits
            generated_tokens, logits = self.generate_from_embeddings(combined_embeds)

            # Decode generated text
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(generated_texts)
            all_generated_texts.append(generated_texts)
            all_logits.append(logits)

        if labels is not None:
            batch_size = len(subgraph_nodes) 
            pad_token_id = self.tokenizer.pad_token_id

            loss = self.compute_normalized_loss(all_logits, labels, batch_size, pad_token_id)

            print(f"Normalized Loss: {loss.item()}")
            return all_generated_texts, loss

        return all_generated_texts

def print_cuda_info():
    print(f"已用显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"保留显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)
    sdg = SDG_LLM('/home/wangjingchu/code/SDGLM/llm/Llama-2-7b-chat-hf').to(device)
    subgraph_nodes = torch.load('/home/wangjingchu/code/SDGLM/TAGDataset/cora/subgraph_nodes.pt')
    se_output = torch.load('/home/wangjingchu/code/SDGLM/structure_encoder/output_cora_tag_pt_module.pt').to(device)
    cora_dataset = torch.load('/home/wangjingchu/code/SDGLM/TAGDataset/cora/cora_tag.pt')
    data = cora_dataset[0]
    labels = sdg.tokenizer([data.label[data.label_map[i]] for i in range(len(data.label_map))], 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True)['input_ids'].to(device)
    labels = labels[:, 1:]

    subgraphs = SubgraphDataset(subgraph_nodes[:140], labels[:140])
    dataloader = DataLoader(subgraphs, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

    inst = {
    'head': "Graph description: Each node represents a paper, edges represent citations. Below are the textual and structural information of target node and the other nodes within the subgraph of the target node.\n",
    'tail': "Categories: Rule Learning, Neural Networks, Case-Based, Genetic Algorithms, Theory, Reinforcement Learning, Probabilistic Methods.\n Please choose the category of the target node from the category list. Please only answer the category's name.\nAnswer: "
    }

    optimizer = torch.optim.Adam(sdg.parameters(), lr=1e-3)

    print_cuda_info()

    epochs = 500
    for i in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            subgraph_nodes = batch['subgraph_nodes']
            labels = batch['labels']
            print(subgraph_nodes, labels)
            _, loss = sdg.batch_generate_and_compute_loss(data.x, se_output, subgraph_nodes, inst, labels)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch {i+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.6f}")

        print(f"Epoch {i+1} completed. Average Loss: {epoch_loss / len(dataloader):.6f}")

        if i % 1 == 0:
            save_path = f'./chkps/sdg_llm_epoch_{i}.pt'
            torch.save({
                'model_state_dict': sdg.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': i,
                'loss': loss.item(),
            }, save_path)
            print(f'Model saved to {save_path}')

if __name__ == '__main__':
    main()