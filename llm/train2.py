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

from transformers import Trainer
import os
import torch

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

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save model checkpoint only every k epochs and only projector parameters.
        """
        # Check if current epoch is a save point
        current_epoch = int(self.state.epoch)
        save_interval = getattr(self.args, "save_per_epochs", 1)  # Default save every epoch
        if True: # current_epoch % save_interval == 0:
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-epoch-{current_epoch}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Save only projector parameters
            projector_state = {
                k: v.cpu() for k, v in model.model.struct_projector.state_dict().items()
            }
            print(f'projector weight shape: {model.model.struct_projector.weight.shape}')

            if self.args.local_rank in [-1, 0]:  # Save on main process only
                os.makedirs(output_dir, exist_ok=True)
                torch.save(projector_state, os.path.join(output_dir, "projector.bin"))
                self.model.config.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Prevent default saving behavior for the full model.
        """
        pass



def sdg_train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # import deepspeed
    # with deepspeed.zero.Init():
    pretrained_config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    dataset = torch.load(data_args.data_path)
    sdg_config = SDGConfig(
        se_dim_in=dataset.struct_encodes.size(1),
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
    model.model.set_struct_projector(proj=None, dim_in=sdg_config.se_dim_in, dim_out=sdg_config.hidden_size)
    print('model done')

    model.is_parallelizable = True
    model.model_parallel = True
    model.config.use_cache = False

    if model_args.freeze_llm_backbone:
        for name, param in model.named_parameters():
            if "struct_projector" not in name:
                param.requires_grad = False

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
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    from torch.utils.data import Subset

    split = dataset.split 
    train_indices = list(range(0, split))
    eval_indices = list(range(split, len(dataset)))

    train_dataset = Subset(dataset, train_indices)
    eval_dataset = Subset(dataset, eval_indices)

    trainer = SDGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    print(f'is_model_parallizable: {trainer.is_model_parallel}')

    print('trainer set done, start training')

    trainer.train()

if __name__ == "__main__":
    sdg_train()