import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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

from sdg_dataset import SDGDataset, MergedSDGDataset
from sdgllama_modeling4 import SDGLlamaForCausalLM, SDGConfig

new_params = ['gpsemlp', 'struct_projector', 'semantic_projector', 'sims_projector', 'graph_token_embedding', 'text_token_embedding', 'sba_temp']

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="./Llama-2-7b-chat-hf")
    struct_proj_path: Optional[str] = field(default='./models_and_data/struct_proj_256.pt')
    gpsemlp_path: Optional[str] = field(default='./models_and_data/cora_gpse_256.pt')
    semantic_path: Optional[str] = field(default=None)
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
    output_dir: str = field(default="./ckpt_tuning_gpse28")
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
    # save_steps: int = field(default=10)

from transformers import Trainer
import os
import torch


from torch_geometric.data import Batch, Data
from transformers.data.data_collator import torch_default_data_collator

def custom_data_collator(features):
    """
    自定义 data_collator, 专门处理包含 torch_geometric.data.Data 类型的 graph 数据。
    """
    for f in features:
        if 'graph' not in f:
            raise ValueError(f"Missing 'graph' in sample: {f}")

    graphs = [f['graph'] for f in features] 
    other_data = [{k: v for k, v in f.items() if k != 'graph'} for f in features] 

    batched_graphs = Batch.from_data_list(graphs)

    batched_other_data = torch_default_data_collator(other_data)

    batched_other_data['graph'] = batched_graphs
    return batched_other_data


class SDGTrainer(Trainer):
    # def get_train_dataloader(self):
    #     """
    #     Overwrites the train DataLoader to ensure batch_size is always 1.
    #     """
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")
        
    #     # Set batch_size to 1
    #     return torch.utils.data.DataLoader(
    #         self.train_dataset,
    #         batch_size=self._train_batch_size,
    #         shuffle=True,  # Shuffle to prevent same-order training
    #         collate_fn=None,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #     )

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
            # # 如果出现Value警告,可能是由于stage3_gather_16bit_weights_on_model_save=false导致的
            # # 这种情况下,将保存完整的检查点,并提示使用zero_to_fp32.py脚本恢复权重
            #     # logger.warning(
            #     #     " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
            #     #     " zero_to_fp32.py to recover weights"
            #     # )
            #     if self.args.should_save:
            #         self._save(output_dir, state_dict={})
            #     # 移除之前的state_dict文件
            #     remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
            #     self.model_wrapped.save_checkpoint(output_dir)  # 保存完整的检查点

        # elif self.args.should_save:
        #     self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        pass
        # if state_dict is None:
        #     pass
        # SAFE_WEIGHTS_NAME = "model.safetensors"
        # WEIGHTS_NAME = "pytorch_model.bin"
        # output_dir = output_dir if output_dir is not None else self.args.output_dir
        # # 创建输出目录(如果不存在)
        # os.makedirs(output_dir, exist_ok=True)
        # # logger.info(f"Saving model checkpoint to {output_dir}")

        # if self.args.save_safetensors:
        #     # 如果设置了save_safetensors,则使用safetensors库保存state_dict
        #     import safetensors
        #     safetensors.torch.save_file(
        #         state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
        #     )
        # else:
        #     # 否则使用torch.save保存state_dict
        #     torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))




def sdg_train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # import deepspeed
    # with deepspeed.zero.Init():
    pretrained_config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    dataset = torch.load(data_args.data_path)
    sdg_config = SDGConfig(
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
    model = SDGLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=sdg_config,
            torch_dtype=dtype
    )

    # sd = torch.load('/home/wangjingchu/code/SDGLM/llm/ckpt_tuning_gpse2/checkpoint-42120/pytorch_model.bin')

    # model_state_dict = model.state_dict()
    # model_state_dict.update(sd)
    # model.load_state_dict(model_state_dict)

    # struct_proj_sd = {}
    # gpsemlp_sd = {}
    # for key, value in sd.items():
    #     if "gpsemlp." in key:
    #         new_key = key.replace("gpsemlp.", "")
    #         gpsemlp_sd[new_key] = value
    #     elif "model.struct_projector." in key:
    #         new_key = key.replace("model.struct_projector.", "")
    #         struct_proj_sd[new_key] = value

    # model.model.set_struct_projector(proj_path='./struct_projector_1024.pt')
    # # model.model.set_semantic_projector(path=model_args.semantic_path)
    # model.set_gpsemlp(gpsemlp_path=model_args.gpsemlp_path)

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

    # from torch.utils.data import Subset

    # split = dataset.split 
    # train_indices = list(range(0, split))
    # eval_indices = list(range(split, len(dataset)))

    # train_dataset = Subset(dataset, train_indices)
    # eval_dataset = Subset(dataset, eval_indices)

    train_dataset = dataset

    trainer = SDGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=custom_data_collator
    )
    trainer.args.save_only_model = True
    # for n, param in trainer.model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(lambda grad, n=n, p=param: print(f"Model Grad for {n} {p.shape}: {grad}, p = {p}"))

    print(f'is_model_parallizable: {trainer.is_model_parallel}')

    print('trainer set done, start training')

    trainer.train()

if __name__ == "__main__":
    sdg_train()
