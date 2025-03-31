import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2Config ,Qwen2DecoderLayer, Qwen2RMSNorm
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
import torch.nn.init as init
from torch_geometric.data import Data
from transformers.utils import logging
from transformers.utils import _prepare_4d_causal_attention_mask_with_cache_position
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.cache_utils import StaticCache, DynamicCache
from gpse_mlp import GPSE_MLP

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "Qwen2Config"


class SDGQwen2Config(Qwen2Config):
    model_type = "sdg"
    def __init__(self, se_dim_in=1024, proj_path=None, gpsemlp_path=None, semantic_path=None,
                    has_gpsemlp=True, has_struct_proj=True, has_semantic_proj=True, n=2, **kwargs):
        super().__init__(**kwargs)
        self.se_dim_in = se_dim_in
        self.proj_path = proj_path
        self.gpsemlp_path = gpsemlp_path
        self.semantic_path = semantic_path
        self.has_gpsemlp = has_gpsemlp
        self.has_struct_proj = has_struct_proj
        self.has_semantic_proj = has_semantic_proj
        self.n = n


class SDGQwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """
    config_class = SDGQwen2Config
    def __init__(self, config: SDGQwen2Config):
        super(SDGQwen2Model, self).__init__(config)
        self.n = config.n
        self.sba_temp = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        self.struct_projector = None
        if config.has_struct_proj:
            self.set_struct_projector(config.proj_path, config.se_dim_in, config.hidden_size)
        self.semantic_projector = None
        if config.has_semantic_proj:
            self.set_semantic_projector(config.semantic_path, config.hidden_size)

    def init_struct_proj(self):
        init.kaiming_normal_(self.struct_projector.weight)

    def set_struct_projector(self, proj_path=None, dim_in=256, dim_out=256):
        n = self.n
        self.struct_projector = nn.ModuleList([nn.Linear(dim_in, dim_out, bias=False) for _ in range(n)])
        if proj_path is not None:
            struct_projector = torch.load(proj_path, map_location='cpu')
            if isinstance(struct_projector, nn.Linear):
                for i in range(n):
                    self.struct_projector[i] = struct_projector.to(self.device)
            elif isinstance(struct_projector, nn.ModuleList):
                self.struct_projector = struct_projector.to(self.device)
            else: 
                raise ValueError("Can't load struct_projector")
        self.struct_projector.to(self.device)
        # else:
        #     self.struct_projector = nn.Linear(dim_in, dim_out, bias=False)

    def set_semantic_projector(self, path=None, dim=4096):
        n = self.n
        self.semantic_projector = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n)])
        if path is not None:
            semantic_projector = torch.load(path, map_location='cpu')
            if isinstance(semantic_projector, nn.Linear):
                for i in range(n):
                    self.semantic_projector[i] = semantic_projector.to(self.device)
            elif isinstance(semantic_projector, nn.ModuleList):
                self.semantic_projector = semantic_projector.to(self.device)
            else: 
                raise ValueError("Can't load semantic_projector")
        self.semantic_projector.to(self.device)

            
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
        # for i in range(len(weighted_t)):
        #     weighted_t[i, i] = 0
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
        mode: Optional[str] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for (i, decoder_layer) in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            n = 0
        num_layers = len(self.layers)
        for i, decoder_layer in enumerate(self.layers):
            # if i==10: break
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

            if i == num_layers//3 or i == num_layers//3*2 :
                use_sims = True
                seq_length = hidden_states.size(1)
                if node_info_mask is None or len(node_info_mask) != seq_length or seq_length==1:
                    node_info_mask = torch.zeros(seq_length, dtype=torch.bool).to(self.device)
                    use_sims = False
                if use_sims:
                    # print('use_sims')
                    if mode == 'node':
                        if sims is None:
                            if struct_encode is None:
                                batch_size = len(input_ids)
                                sims = torch.eye(batch_size, device=self.device)
                            elif use_struct_projector: 
                                if self.struct_projector is None:
                                    raise ValueError("struct_projector is None")
                                # print('use_struct')
                                struct_encode = self.struct_projector[n](struct_encode)
                                sims = self.sim(struct_encode, struct_encode)
                            else:
                                sims = self.sim(struct_encode, struct_encode)
                        subgraph_nodes = subgraph_nodes if subgraph_nodes is not None else torch.ones((struct_encode.size(0), struct_encode.size(0)))
                        node_info_mask = node_info_mask.bool()
                        node_info_mask = node_info_mask.unsqueeze(0).expand(hidden_states.size(0), -1)
                        node_info = hidden_states[node_info_mask].view(hidden_states.size(0), -1, hidden_states.size(-1))
                        neigh_info = self.structure_attention(node_info, sims, subgraph_nodes, self.sba_temp)
                        (batch_size, info_len, hid_dim) = node_info.size()

                        update_info = neigh_info
                        if use_semantic_proj:
                            if self.semantic_projector is None:
                                raise ValueError("semantic_projector is None")
                            # print('use_sematic')
                            update_info = self.semantic_projector[n](update_info)
                 
                    elif mode=='edge':
                        if sims is None:
                            if struct_encode is None:
                                batch_size = len(input_ids)
                                sims1 = torch.eye(batch_size, device=self.device)
                                sims2 = torch.eye(batch_size, device=self.device)
                            elif use_struct_projector: 
                                if self.struct_projector is None:
                                    raise ValueError("struct_projector is None")
                                struct_encode1 = self.struct_projector[n](struct_encode[0])
                                struct_encode2 = self.struct_projector[n](struct_encode[1])
                                sims1 = self.sim(struct_encode1, struct_encode1)
                                sims2 = self.sim(struct_encode2, struct_encode2)
                            else:
                                sims1 = self.sim(struct_encode[0], struct_encode[0])
                                sims2 = self.sim(struct_encode[1], struct_encode[1])
                        subgraph_nodes = subgraph_nodes if subgraph_nodes is not None else torch.ones((2, len(input_ids), len(input_ids)))
                        node_info_mask = node_info_mask.bool()
                        node_info_mask = node_info_mask.unsqueeze(0).expand(hidden_states.size(0), -1)
                        node_info = hidden_states[node_info_mask].view(hidden_states.size(0), -1, hidden_states.size(-1))
                        node_info_len = node_info.size(1)
                        node_info1 = node_info[:, :node_info_len//2, :]
                        node_info2 = node_info[:, node_info_len//2:, :]
                        neigh_info1 = self.structure_attention(node_info1, sims1, subgraph_nodes[0], self.sba_temp)
                        neigh_info2 = self.structure_attention(node_info2, sims2, subgraph_nodes[1], self.sba_temp)
                        (batch_size, info_len, hid_dim) = node_info1.size()

                        update_info1 = neigh_info1
                        update_info2 = neigh_info2
                        if use_semantic_proj:
                            if self.semantic_projector is None:
                                raise ValueError("semantic_projector is None")
                            update_info1 = self.semantic_projector[n](update_info1)
                            update_info2 = self.semantic_projector[n](update_info2)
                        update_info = torch.cat([update_info1, update_info2], dim=1)

                    update_info = update_info + node_info
                    expanded_mask = node_info_mask.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)) 
                    hidden_states.masked_scatter_(expanded_mask, update_info.view(-1, hid_dim))
                    n += 1 

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



class SDGQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(SDGQwen2ForCausalLM, self).__init__(config)
        self.model = SDGQwen2Model(config)
        self.gpsemlp = None
        if config.has_gpsemlp:
            self.set_gpsemlp(config.gpsemlp_path)

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        struct_encode: torch.LongTensor = None,
        subgraph_nodes: torch.Tensor = None,
        x: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        edge_index2: Optional[torch.Tensor] = None,
        graph: Optional[Data] = None,
        valid_nodes_mask: Optional[torch.Tensor] = None,
        node_info_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sims: Optional[torch.Tensor] = None,
        mode: Optional[int] = 0,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mode = 'edge' if mode == 1 else 'node'
        if use_gpsemlp or use_struct_projector:
            sims = None
            
        g_loss = 0
        x = x.squeeze(0) if x is not None else None
        edge_index = edge_index.squeeze(0) if edge_index is not None else None
        edge_index2 = edge_index.squeeze(0) if edge_index is not None else None
        if (graph is not None or x is not None) and use_gpsemlp:
            if self.gpsemlp is None:
                raise ValueError("gpse is None, can't use gpse")
            if mode=='node':
                graph = Data(x=x, edge_index=edge_index).to(self.device)
                struct_encode = self.gpsemlp(graph)
                g_loss += self.gpsemlp.constractive_loss(graph, struct_encode)
            elif mode=='edge':
                g1 = Data(x=x, edge_index=edge_index).to(self.device)
                g2 = Data(x=x, edge_index=edge_index2).to(self.device)
                s1 = self.gpsemlp(g1)
                s2 = self.gpsemlp(g2)
                struct_encode = [s1.tolist(), s2.tolist()]
                struct_encode = torch.tensor(struct_encode).to(self.device)
                g_loss += (self.gpsemlp.constractive_loss(g1, s1) + self.gpsemlp.constractive_loss(g2, s2)) / 2

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            struct_encode=struct_encode,
            subgraph_nodes=subgraph_nodes,
            valid_nodes_mask=valid_nodes_mask,
            node_info_mask=node_info_mask,
            attention_mask=attention_mask,
            sims=sims,
            mode=mode,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            use_struct_projector=use_struct_projector,
            use_semantic_proj=use_semantic_proj,
            use_sims_proj=use_sims_proj,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.squeeze(0)
            if valid_nodes_mask is None:
                valid_nodes_mask = torch.ones((input_ids.size(0), input_ids.size(0)))
            valid_indices = torch.nonzero(valid_nodes_mask == 1, as_tuple=True)[0]
            if mode=='edge': valid_indices = torch.tensor([0])
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

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        struct_encode=None, 
        subgraph_nodes=None, 
        graph=None, 
        valid_nodes_mask=None, 
        node_info_mask=None, 
        sims=None, 
        mode=0, 
        x=None, 
        edge_index=None, 
        edge_index2=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "struct_encode": struct_encode,
                "subgraph_nodes": subgraph_nodes,
                "valid_nodes_mask": valid_nodes_mask,
                "sims": sims,
                "node_info_mask": node_info_mask,
                "graph": graph,
                "mode": mode,
                "x": x,
                "edge_index": edge_index,
                "edge_index2": edge_index2,
            }
        )
        return model_inputs