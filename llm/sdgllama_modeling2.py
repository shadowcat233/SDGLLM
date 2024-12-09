import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import LlamaModel, LlamaForCausalLM, AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
# from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaModel, LlamaDecoderLayer, \
    logger, BaseModelOutputWithPast, Cache, DynamicCache, _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask, CausalLMOutputWithPast, LlamaForCausalLM
from typing import List, Optional, Tuple, Union
import torch.nn.init as init
from torch_geometric.data import Data

from gpse_mlp import GPSE_MLP

class SDGConfig(LlamaConfig):
    model_type = "sdg"
    def __init__(self, se_dim_in=1024, proj_path=None, gpsemlp_path=None, **kwargs):
        super().__init__(**kwargs)
        self.se_dim_in = se_dim_in
        self.proj_path = proj_path
        self.gpsemlp_path = gpsemlp_path



class SDGLlamaModel(LlamaModel):
    config_class = SDGConfig

    def __init__(self, config: SDGConfig):
        super(SDGLlamaModel, self).__init__(config)
        self.set_struct_projector(config.proj_path, config.se_dim_in, config.hidden_size)
        # print(self.struct_projector.state_dict())

    def init_struct_proj(self):
        init.kaiming_normal_(self.struct_projector.weight)


    def set_struct_projector(self, proj_path=None, dim_in=1024, dim_out=4096):
        if proj_path is not None:
            self.struct_projector = torch.load(proj_path)
        else:
            self.struct_projector = nn.Linear(dim_in, dim_out, bias=False)
            

    def sim(self, z1, z2):
        z1 = F.normalize(z1) 
        z2 = F.normalize(z2) 
        return torch.mm(z1, z2.t()) 

    def structure_attention(self, t, sims, sg_nodes, temp=0.4):
        f = lambda x: torch.exp(x/temp)
        filtered_sims = f(sims) * sg_nodes
        sims_expanded = filtered_sims.unsqueeze(-1).unsqueeze(-1) 
        weighted_t = (sims_expanded * t.unsqueeze(1))
        sim_sum = filtered_sims.sum(dim=1, keepdim=True).unsqueeze(-1) 
        sim_sum = sim_sum + (sim_sum == 0).float() 
        t_hat = (weighted_t.sum(dim=0) / sim_sum).squeeze(-1)
        return t_hat
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        struct_encode: torch.LongTensor = None,
        subgraph_nodes: torch.Tensor = None,
        valid_nodes_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if self.device == torch.device('cuda:0'):
        # print(self.struct_projector.state_dict())
        attention_mask = attention_mask.squeeze(0)
        input_ids = input_ids.squeeze(0)
        struct_encode = struct_encode.squeeze(0)
        subgraph_nodes = subgraph_nodes.squeeze(0)
        valid_nodes_mask = valid_nodes_mask.squeeze(0)

        se = self.struct_projector(struct_encode)
        sims = self.sim(se, se)
        subgraph_nodes = subgraph_nodes if subgraph_nodes is not None else torch.ones((se.size(0), se.size(0)))

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if i > 0 and i % 8 == 0 and struct_encode is not None:
                # if self.device == torch.device('cuda:0'): print(i)
                hidden_states = self.structure_attention(hidden_states, sims, subgraph_nodes)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class SDGLlamaForCausalLM(LlamaForCausalLM):
    config_class = SDGConfig
    def __init__(self, config):
        super(SDGLlamaForCausalLM, self).__init__(config)
        self.model = SDGLlamaModel(config)
        self.set_gpsemlp(config.gpsemlp_path)
        
    def set_gpsemlp(self, gpsemlp_path):
        self.gpsemlp = torch.load(gpsemlp_path).to(self.device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        struct_encode: torch.LongTensor = None,
        subgraph_nodes: torch.Tensor = None,
        graph: Data = None,
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

        g_loss = 0
        if graph is not None:
            struct_encode = self.gpsemlp(graph)
            g_loss += self.gpsemlp.constractive_loss(graph, struct_encode)

        # print(input_ids.shape)

        # if labels is None:
        #     labels = input_ids.squeeze(0)

        outputs = self.model(
            input_ids=input_ids,
            struct_encode=struct_encode,
            subgraph_nodes=subgraph_nodes,
            valid_nodes_mask=valid_nodes_mask,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states.to(dtype=self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            if valid_nodes_mask is None:
                valid_nodes_mask = torch.ones((input_ids.size(0), input_ids.size(0)))
            valid_indices = torch.nonzero(valid_nodes_mask == 1, as_tuple=True)[0]
            valid_cnt = valid_nodes_mask.sum()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_logits = shift_logits[valid_indices]
            shift_labels = shift_labels[valid_indices]
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            seq_len = input_ids.squeeze(0).size(1)
            loss = loss / (valid_cnt * seq_len)
            loss += g_loss * 0.1
            
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
        self, input_ids, struct_encode, subgraph_nodes, valid_nodes_mask, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds,
                            "struct_encode": struct_encode,
                            "subgraph_nodes": subgraph_nodes,
                            "valid_nodes_mask": valid_nodes_mask}
        else:
            model_inputs = {"input_ids": input_ids,
                            "struct_encode": struct_encode,
                            "subgraph_nodes": subgraph_nodes,
                            "valid_nodes_mask": valid_nodes_mask}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("sdg", SDGConfig)
AutoModelForCausalLM.register(SDGConfig, SDGLlamaForCausalLM)