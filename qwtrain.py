import os

import transformers
from transformers import HfArgumentParser, Trainer, AutoConfig, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from torch import nn
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
upper_dir = os.path.dirname(current_dir)
sys.path.append(upper_dir)

from sdg_dataset import SDGDataset, MergedSDGDataset, SDGEdgeDataset
from modeling_sdgqwen import SDGQwen2Config, SDGQwen2ForCausalLM

new_params = ['gpsemlp', 'struct_projector', 'semantic_projector', 'sims_projector', 'graph_token_embedding', 'text_token_embedding', 'sba_temp']

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Llama-2-7b-chat-hf")
    struct_proj_path: Optional[str] = field(default='../models_and_data/struct_proj_n2_c_all.pt')
    gpsemlp_path: Optional[str] = field(default='../models_and_data/gpsemlp_c_all.pt')
    semantic_path: Optional[str] = field(default='../models_and_data/semantic_proj_c_all.pt')
    # sims_path: Optional[str] = field(default='../structure_encoder/sims_proj.pt')
    version: Optional[str] = field(default="v0")
    freeze_llm_backbone: bool = field(default=True)
    sa_layer_nums: int = field(default=1)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_dropout: float = field(default=0.05)

@dataclass
class DataArguments:
    data_path: str = field(default="../merged_few_shot_sdg_apw.pt")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./ckpt_tuning_gpse50")
    deepspeed: str = field(default="./deepspeed_config.json")
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    logging_steps: int = field(default=10)
    logging_first_step: bool = field(default=True)
    # load_best_model_at_end: bool = field(default=True)
    learning_rate: float = field(default=2e-3)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    save_steps: int = field(default=500)
    num_train_epochs: int = field(default=5)

from transformers import Trainer
import os
import torch


from torch_geometric.data import Batch, Data
from transformers.data.data_collator import torch_default_data_collator

def custom_data_collator(features):
    """
    自定义 data_collator, 专门处理包含 torch_geometric.data.Data 类型的 graph 数据。
    """
    other_data = [{k: v for k, v in f.items()} for f in features] 
    batched_other_data = torch_default_data_collator(other_data)

    return batched_other_data


class SDGTrainer(Trainer):

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model checkpoint only every k epochs and only projector parameters.
        """
        WEIGHTS_NAME = "pytorch_model.bin"
        if self.is_deepspeed_enabled:
        # 如果启用了DeepSpeed,则需要使用self.accelerator.get_state_dict获取模型状态字典
        # .get_state_dict 这个函数可以将ZeRO3切片在其他设备上的参数加载过来
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if state_dict is not None:
                    state_dict = {k: v for k, v in state_dict.items() if any(x in k for x in new_params)}
                    if self.args.should_save:
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            except ValueError:
                pass


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass




def sdg_train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # import deepspeed
    # with deepspeed.zero.Init():
    pretrained_config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    dataset = torch.load(data_args.data_path)
    sdg_config = SDGQwen2Config(
        se_dim_in=256,
        proj_path=model_args.struct_proj_path,
        gpsemlp_path=model_args.gpsemlp_path,
        semantic_path=model_args.semantic_path,
        # sims_path=model_args.sims_path,
        # has_gpsemlp=False, 
        # has_struct_proj=False,
        # has_semantic_proj=False,
        # has_sims_proj=True,
        **pretrained_config.to_dict()
    )
    print('sdg_config done')
    dtype = torch.float16 if training_args.fp16 else torch.float
    dtype = torch.bfloat16 if training_args.bf16 else dtype
    model = SDGQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=sdg_config,
            torch_dtype=dtype
    )

    print('model done')

    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False

    if model_args.freeze_llm_backbone:
        for n, param in model.named_parameters():
            if any(x in n for x in new_params):
                param.requires_grad = True
            else: param.requires_grad = False

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM" 
        )
        model = get_peft_model(model, lora_config)
        print(model)
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    from torch.utils.data import Subset

    # split = dataset.split 
    # split = 10
    # train_indices = list(range(0, split))
    # eval_indices = list(range(split, len(dataset)))

    # train_dataset = Subset(dataset, train_indices)
    # eval_dataset = Subset(dataset, eval_indices)

    train_dataset = dataset
    training_args.save_steps = len(train_dataset)

    trainer = SDGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=custom_data_collator
    )
    trainer.args.save_only_model = True
    # for n, param in trainer.model.named_parameters():
    #     if param.requires_grad and ('semantic' in n):
    #         param.register_hook(lambda grad, n=n, p=param: print(f"Model Grad for {n} {p.shape}: {grad}, p = {p}"))

    print(f'is_model_parallizable: {trainer.is_model_parallel}')

    print('trainer set done, start training')

    trainer.train()

if __name__ == "__main__":
    sdg_train()