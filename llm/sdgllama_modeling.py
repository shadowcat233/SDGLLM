import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import LlamaModel, LlamaForCausalLM, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union


class SDGConfig(LlamaConfig):
    model_type = "sdg"
    def __init__(self, se_dim_in, sa_layer_nums, **kwargs):
        super().__init__(**kwargs)
        self.se_dim_in = se_dim_in
        self.sa_layer_nums = sa_layer_nums



class SDGLlamaModel(LlamaModel):
    config_class = SDGConfig

    def __init__(self, config: SDGConfig):
        super(SDGLlamaModel, self).__init__(config)

class SDGLlamaForCausalLM(LlamaForCausalLM):
    config_class = SDGConfig
    def __init__(self, config):
        super(SDGLlamaForCausalLM, self).__init__(config)
        self.model = SDGLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.projector = nn.Linear(config.se_dim_in, config.hidden_size)
        self.sa_layer_nums = config.sa_layer_nums

        # self.post_init()
    
    def get_model(self):
        return self.model
    
    def sim(self, z1, z2):
        z1 = F.normalize(z1) 
        z2 = F.normalize(z2) 
        return torch.mm(z1, z2.t()) 

    def structure_attention(self, t, sims, sg_nodes):

        filtered_sims = sims * sg_nodes
        sims_expanded = filtered_sims.unsqueeze(-1).unsqueeze(-1) 
        weighted_t = (sims_expanded * t.unsqueeze(1))
        sim_sum = filtered_sims.sum(dim=1, keepdim=True).unsqueeze(-1) 
        sim_sum = sim_sum + (sim_sum == 0).float() 
        t_hat = (weighted_t.sum(dim=1) / sim_sum).squeeze(-1)
        return t_hat


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        struct_encode: torch.LongTensor = None,
        subgraph_nodes: torch.Tensor = None,
        valid_nodes_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else True

        attention_mask = attention_mask.squeeze(0)
        input_ids = input_ids.squeeze(0)
        struct_encode = struct_encode.squeeze(0)
        subgraph_nodes = subgraph_nodes.squeeze(0)
        valid_nodes_mask = valid_nodes_mask.squeeze(0)
        print(attention_mask.shape, input_ids.shape, struct_encode.shape, subgraph_nodes.shape, valid_nodes_mask.shape)

        if labels is None:
            labels = input_ids.squeeze(0)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        # del outputs
        # torch.cuda.empty_cache()
        # past_key_values = outputs.past_key_values

        if struct_encode is not None:
            se = self.projector(struct_encode)
            sims = self.sim(se, se)
            subgraph_nodes = subgraph_nodes if subgraph_nodes is not None else torch.ones((se.size(0), se.size(0)))

            for i in range(self.sa_layer_nums):
                hidden_states = self.structure_attention(hidden_states, sims, subgraph_nodes)
                print(hidden_states.shape, attention_mask.shape)
                # outputs = self.model(
                #     attention_mask=attention_mask,
                #     past_key_values=past_key_values,
                #     inputs_embeds=hidden_states,
                #     use_cache=use_cache,
                #     output_attentions=output_attentions,
                #     output_hidden_states=output_hidden_states,
                #     return_dict=return_dict
                # )
                # hidden_states = outputs[0]
                # # past_key_values = outputs.past_key_values
                # if i<self.sa_layer_nums-1:
                #     del outputs
                #     torch.cuda.empty_cache()
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if valid_nodes_mask is None:
                valid_nodes_mask = torch.ones((input_ids.size(0), input_ids.size(0)))
            valid_indices = torch.nonzero(valid_nodes_mask == 1, as_tuple=True)[0]
            valid_cnt = valid_nodes_mask.sum()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_logits = shift_logits[valid_indices]
            shift_labels = shift_labels[valid_indices]
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            seq_len = input_ids.size(0)
            loss = loss / (valid_cnt * seq_len)
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "struct_encode": kwargs.get("struct_encode", None),
                "subgraph_nodes": kwargs.get("subgraph_nodes", None),
            }
        )
        return model_inputs

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("sdg", SDGConfig)
AutoModelForCausalLM.register(SDGConfig, SDGLlamaForCausalLM)