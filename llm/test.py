from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from sdgllama_modeling2 import SDGLlamaForCausalLM, SDGConfig
import torch

pretrained_config = LlamaConfig.from_pretrained('./Llama-2-7b-chat-hf')

sdg_config = SDGConfig(
    se_dim_in=1024,
    **pretrained_config.to_dict()
)

model = SDGLlamaForCausalLM.from_pretrained(
        './Llama-2-7b-chat-hf',
        config=sdg_config,
        torch_dtype=torch.float16
)

estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=5, num_nodes=1)