import os


import transformers
from transformers import HfArgumentParser, Trainer, AutoConfig, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch import nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import tqdm
import torch

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)

from sdg_dataset import SDGDataset
from sdgllama_modeling2 import SDGLlamaForCausalLM, SDGConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Llama-2-7b-chat-hf")
    version: Optional[str] = field(default="v0")
    freeze_llm_backbone: bool = field(default=True)
    sa_layer_nums: int = field(default=1)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_dropout: float = field(default=0.05)

@dataclass
class DataArguments:
    data_path: str = field(default="../cora_sdg_dataset.pt")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./checkpoints")
    deepspeed: str = field(default="./deepspeed_config.json")
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    logging_steps: int = field(default=10)
    logging_first_step: bool = field(default=True)
    # load_best_model_at_end: bool = field(default=True)
    learning_rate: float = field(default=1e-4)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    save_per_ckpts: int = field(default=1)


ds_config = {
  "fp16": {
      "enabled": False,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "initial_scale_power": 16,
      "hysteresis": 2,
      "min_loss_scale": 1
  },

  "optimizer": {
      "type": "AdamW",
      "params": {
          "lr": 2e-4
      },
      "zero_force_ds_cpu_optimizer": False
  },

  "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 1e-7,
          "warmup_max_lr": 2e-4,
          "warmup_num_steps": 10
      }
  },

  "zero_optimization": {
      "stage": 3,
      "offload_optimizer": {
          "device": "cpu",
          "pin_memory": True
      },
      "offload_param": {
          "device": "cpu",
          "pin_memory": True
      },
      "overlap_comm": True,
      "contiguous_gradients": True,
      "sub_group_size": 1e9,
      "stage3_max_live_parameters": 1e9,
      "stage3_max_reuse_distance": 1e9,
      "stage3_gather_16bit_weights_on_model_save": True
  },
#   "zero_optimization": {
#     "stage": 2,
#     "offload_optimizer": {
#       "device": "cpu",
#       "pin_memory": True
#     },
#     "allgather_partitions": True,
#     "allgather_bucket_size": 5e8,
#     "overlap_comm": True,
#     "reduce_scatter": True,
#     "reduce_bucket_size": 5e8,
#     "contiguous_gradients": True,
#     "round_robin_gradients": True
#   },
#     "zero_optimization": {
#     "stage": 1
#   },

  "gradient_accumulation_steps": 1,
  "steps_per_print": 100,
  "train_micro_batch_size_per_gpu": 1,
  "wall_clock_breakdown": False
}

def sdg_train():
    # parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')
    dataset = torch.load("../cora_sdg_dataset.pt")
    sdg_config = SDGConfig(
        se_dim_in=dataset.struct_encodes.size(1),
        **pretrained_config.to_dict()
    )
    print('sdg_config done')
    # dtype = torch.float16 if training_args.fp16 else torch.float
    # dtype = torch.bfloat16 if training_args.bf16 else dtype
    model = SDGLlamaForCausalLM.from_pretrained(
            './Llama-2-7b-chat-hf',
            config=sdg_config,
            torch_dtype=torch.float
    )

    for name, param in model.named_parameters():
        if "weight" in name and ("31" in name or "struct" in name):
            print(name, param.shape)

    proj = torch.load(f'./checkpoints/struct_proj.pt')
    model.model.set_struct_projector(proj=proj)
    print('model done')
    print("Struct proj parameters:", list(model.model.struct_projector.parameters()))

    for name, param in model.named_parameters():
        if "weight" in name and ("31" in name or "struct" in name):
            print(name, param.shape)

    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False

    # opt = DeepSpeedCPUAdam(model.parameters(),lr=2e-4)
    # print("Optimizer params:", opt.param_groups)

    if True:
        model.requires_grad_(False)
        for param in model.model.struct_projector.parameters():
            param.requires_grad = True
    # print("Struct proj parameters:", list(model.model.struct_projector.parameters()))

    from torch.utils.data import Subset

    split = dataset.split 
    train_indices = list(range(0, split))
    eval_indices = list(range(split, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=1, sampler=DistributedSampler(train_dataset))
    

    # import json
    # with open(training_args.deepspeed, 'r') as f:
    #     deepspeed_config = json.load(f)

    import deepspeed
    model = model.cuda()
    model_engine, optimizer, _, _ = deepspeed.initialize(   args=None, 
                                                            model=model, 
                                                            model_parameters=model.parameters(),
                                                            # optimizer=opt,
                                                            config=ds_config)
    # proj = torch.load(f'./checkpoints/struct_proj.pt')
    # model.model.set_struct_projector(proj=proj)
    # model.model.struct_projector.to(model_engine.local_rank)
    # model_engine.optimizer.optimizer.add_param_group({"params": model.model.struct_projector.parameters()})
    print('---------------------------------------------------------------')
    # print(model_engine.optimizer_name)
    print("Struct proj parameters after ds init:", list(model.model.struct_projector.parameters()))
    print("Optimizer.optimizer params:", model_engine.optimizer.optimizer.param_groups)

    # for name, param in model.named_parameters():
    #     if "weight" in name:
    #         print(name, param.shape)

    # exit()
    model.train()
    for epoch in range(3):
        total_loss = 0
        progress_bar = tqdm.tqdm(train_dataloader, desc="Training", total=len(train_dataloader))
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(model_engine.local_rank) 
            attention_mask = batch['attention_mask'].to(model_engine.local_rank) 
            struct_encode = batch['struct_encode'].to(model_engine.local_rank) 
            subgraph_nodes = batch['subgraph_nodes'].to(model_engine.local_rank) 
            valid_nodes_mask = batch['valid_nodes_mask'].to(model_engine.local_rank) 

            # 前向传播
            outputs = model_engine(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            struct_encode=struct_encode,
                            subgraph_nodes=subgraph_nodes,
                            valid_nodes_mask=valid_nodes_mask,
                            labels=input_ids)
            loss = outputs['loss']
            total_loss += loss.item()

            # 反向传播和优化
            model_engine.backward(loss)
            # print("Optimizer state before step:", model_engine.optimizer.state_dict())
            model_engine.step()
            # print("Optimizer state after step:", model_engine.optimizer.state_dict())


            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

            if (1+idx) % 10 == 0:
                save_dir = f'./checkpoints'
                tag = str(epoch)
                if not os.path.exists(os.path.join(save_dir, tag)):
                    os.makedirs(os.path.join(save_dir, tag))
                model_engine._save_checkpoint(save_dir, tag=tag, exclude_frozen_parameters=True)

        print(f'epoch {epoch} average loss: {(total_loss/len(train_dataloader)) :.4f}')

if __name__ == "__main__":
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    sdg_train()