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
    def __init__(self, se_dim_in=1024, proj_path=None, gpsemlp_path=None, semantic_path=None,
                    has_gpsemlp=True, has_struct_proj=True, has_semantic_proj=True, **kwargs):
        super().__init__(**kwargs)
        self.se_dim_in = se_dim_in
        self.proj_path = proj_path
        self.gpsemlp_path = gpsemlp_path
        self.semantic_path = semantic_path
        self.has_gpsemlp = has_gpsemlp
        self.has_struct_proj = has_struct_proj
        self.has_semantic_proj = has_semantic_proj



class SDGLlamaModel(LlamaModel):
    config_class = SDGConfig

    def __init__(self, config: SDGConfig):
        super(SDGLlamaModel, self).__init__(config)
        self.sba_temp = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        self.struct_projector = None
        if config.has_struct_proj:
            self.set_struct_projector(config.proj_path, config.se_dim_in, config.hidden_size)
        self.semantic_projector = None
        if config.has_semantic_proj:
            self.set_semantic_projector(config.semantic_path, config.hidden_size)

    def init_struct_proj(self):
        init.kaiming_normal_(self.struct_projector.weight)

    def set_struct_projector(self, proj_path=None, dim_in=1024, dim_out=4096):
        if proj_path is not None:
            struct_projector = torch.load(proj_path, map_location='cpu')
            self.struct_projector = struct_projector.to(self.device)
        else:
            self.struct_projector = nn.Linear(dim_in, dim_out, bias=False)

    def set_semantic_projector(self, path=None, dim=4096):
        if path is not None:
            semantic_projector = torch.load(path, map_location='cpu')
            self.semantic_projector = semantic_projector.to(self.device)
        else:
            self.semantic_projector = nn.Linear(dim, dim)
            
    def sim(self, z1, z2):
        # print(z1.shape)
        z1 = F.normalize(z1) 
        z2 = F.normalize(z2) 
        return torch.mm(z1, z2.t()) 

    def structure_attention(self, t, sims, sg_nodes, temp=0.2):
        '''
        t: hidden_states, [batch_size, seq_len, embed_dim]
        sims: 相似度矩阵, [batch_size, batch_size]
        sg_nodes: 子图filter, [i, j]为1表示点j在点i的子图中, [batch_size, batch_size]
        temp: 温度缩放，控制相似度值的平滑程度
        '''
        temp = temp if temp >= 0.05 else 0.05
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
        node_info_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sims: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_struct_projector: Optional[bool] = True,
        use_semantic_proj: Optional[bool] = True,
        use_sims_proj: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if self.device == torch.device('cuda:0'):
        # print(self.struct_projector.state_dict())
        if True or (input_ids.dim() == 4 and input_ids.size(0) == 1):
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            struct_encode = struct_encode.squeeze(0) if struct_encode is not None else None
            subgraph_nodes = subgraph_nodes.squeeze(0) if subgraph_nodes is not None else None 
            valid_nodes_mask = valid_nodes_mask.squeeze(0) if valid_nodes_mask is not None else None
            sims = sims.squeeze(0) if sims is not None else None
            node_info_mask = node_info_mask.squeeze(0) if node_info_mask is not None else None
          
        subgraph_nodes = subgraph_nodes if subgraph_nodes is not None else torch.ones((struct_encode.size(0), struct_encode.size(0)))

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

            if i == 15:
                use_sims = True
                if node_info_mask is None or len(node_info_mask) != seq_length or batch_size == 1:
                    node_info_mask = torch.zeros(seq_length, dtype=torch.bool).to(self.device)
                    use_sims = False
                if use_sims:
                    if sims is None:
                        if struct_encode is None:
                            batch_size = len(input_ids)
                            sims = torch.eye(batch_size, device=self.device)
                        elif use_struct_projector: 
                            if self.struct_projector is None:
                                raise ValueError("struct_projector is None")
                            struct_encode = self.struct_projector(struct_encode)
                            sims = self.sim(struct_encode, struct_encode)
                        else:
                            sims = self.sim(struct_encode, struct_encode)
                    node_info_mask = node_info_mask.bool()
                    # print(node_info_mask)
                    node_info_mask = node_info_mask.unsqueeze(0).expand(hidden_states.size(0), -1)
                    node_info = hidden_states[node_info_mask].view(hidden_states.size(0), -1, hidden_states.size(-1))
                    # print(node_info.shape)
                    neigh_info = self.structure_attention(node_info, sims, subgraph_nodes, self.sba_temp)
                    (batch_size, info_len, hid_dim) = node_info.size()
                    # even_pairs = info_len // 2  
                    # node_info_even = node_info[:, :even_pairs * 2, :].view(batch_size, even_pairs, 2, hid_dim).mean(dim=2)
                    # node_info_last = node_info[:, even_pairs * 2:, :]
                    # node_info_compressed = torch.cat([node_info_even, node_info_last], dim=1)

                    # neigh_info_compressed = neigh_info[:, :even_pairs * 2, :].view(batch_size, even_pairs, 2, hid_dim).mean(dim=2)
                    # if seq_length == 1:
                    #     node_info_compressed = (node_info + neigh_info) / 2

                    # update_info = torch.cat([node_info_compressed, neigh_info_compressed], dim=1)
                    update_info = neigh_info
                    if use_semantic_proj:
                        if self.semantic_projector is None:
                            raise ValueError("semantic_projector is None")
                        update_info = self.semantic_projector(update_info)
                    
                    expanded_mask = node_info_mask.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)) 
                    hidden_states.masked_scatter_(expanded_mask, update_info.view(-1, hid_dim))


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
        self.gpsemlp = None
        if config.has_gpsemlp:
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
        node_info_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sims: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_struct_projector: Optional[bool] = True,
        use_gpsemlp: Optional[bool] = True,
        return_all_hidden_states: Optional[bool] = False,
        use_semantic_proj: Optional[bool] = True,
        use_sims_proj: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else True

        if use_gpsemlp or use_struct_projector:
            sims = None
            
        g_loss = 0
        if graph is not None and use_gpsemlp:
            if self.gpsemlp is None:
                raise ValueError("gpse is None, can't use gpse")
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
            node_info_mask=node_info_mask,
            attention_mask=attention_mask,
            sims=sims,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_struct_projector=use_struct_projector,
            use_semantic_proj=use_semantic_proj,
            use_sims_proj=use_sims_proj,
        )
        hidden_states = outputs['last_hidden_state']
        
        logits = self.lm_head(hidden_states.to(dtype=self.lm_head.weight.dtype))

        loss = None
        if labels is not None:
            labels = labels.squeeze(0)
            if valid_nodes_mask is None:
                valid_nodes_mask = torch.ones((input_ids.size(0), input_ids.size(0)))
            valid_indices = torch.nonzero(valid_nodes_mask == 1, as_tuple=True)[0]
            valid_cnt = valid_nodes_mask.sum()
            node_info_mask = node_info_mask.squeeze(0) if node_info_mask is not None else torch.ones(labels.size(1), dtype=torch.bool)

            logits = logits[valid_indices]
            labels = labels[valid_indices]
            
            # node_info_mask_expanded = node_info_mask.unsqueeze(0).expand(labels.size(0), -1)
            # labels = torch.where(node_info_mask_expanded, 0, labels)

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=0)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            num_zeros = torch.sum(shift_labels == 0).item()

            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            loss = loss + g_loss * 0.1
            
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if return_all_hidden_states:
            hidden_states = outputs['hidden_states']

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, struct_encode=None, subgraph_nodes=None, graph=None, valid_nodes_mask=None, node_info_mask=None, sims=None, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds,
                            "struct_encode": struct_encode,
                            "subgraph_nodes": subgraph_nodes,
                            "valid_nodes_mask": valid_nodes_mask,
                            "sims": sims}
        else:
            model_inputs = {"input_ids": input_ids,
                            "struct_encode": struct_encode,
                            "subgraph_nodes": subgraph_nodes,
                            "valid_nodes_mask": valid_nodes_mask,
                            "sims": sims}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "node_info_mask": node_info_mask,
                "graph": graph,
            }
        )
        return model_inputs

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("sdg", SDGConfig)
AutoModelForCausalLM.register(SDGConfig, SDGLlamaForCausalLM)