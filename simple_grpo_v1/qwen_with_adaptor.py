from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import torch
import torch.nn as nn
from typing import Callable, Optional, Union
import copy

class RolloutHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.layer2 = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

    def zero_init(self):
        with torch.no_grad():
            nn.init.constant_(self.layer1.weight, 0.0)
            if self.layer1.bias is not None:
                nn.init.constant_(self.layer1.bias, 0.0)

            nn.init.constant_(self.layer2.weight, 0.0)
            if self.layer2.bias is not None:
                nn.init.constant_(self.layer2.bias, 0.0)
    
    def random_init(self, std=1e-6):
        with torch.no_grad():
            nn.init.normal_(self.layer1.weight, mean=0.0, std=std)
            if self.layer1.bias is not None:
                nn.init.constant_(self.layer1.bias, 0.0)

            nn.init.normal_(self.layer2.weight, mean=0.0, std=std)
            if self.layer2.bias is not None:
                nn.init.constant_(self.layer2.bias, 0.0)


class QwenWithRolloutHead(Qwen2ForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.rollout_head = RolloutHead(config)
        self.use_adaptor = False

    def reset_adaptor(self):
        self.rollout_head.random_init()

    # def _try_get_tensor_from_weights(self, weights_obj, key):
    #     """
    #     Try multiple ways to get a tensor from vLLM weights loader or a dict.
    #     Returns a torch.Tensor on success, or None if not found.
    #     """
    #     # helper to coerce to torch tensor
    #     def to_tensor(x):
    #         if isinstance(x, torch.Tensor):
    #             return x
    #         try:
    #             return torch.as_tensor(x)
    #         except Exception:
    #             return None

    #     candidates = [
    #         key,
    #         "model." + key,
    #         "transformer." + key,
    #         "qwen." + key,
    #     ]
    #     # If the weights_obj supports get_tensor (safetensors wrapper)
    #     for k in candidates:
    #         try:
    #             if hasattr(weights_obj, "get_tensor"):
    #                 t = weights_obj.get_tensor(k)
    #                 return to_tensor(t)
    #         except Exception:
    #             pass
    #         try:
    #             # dict-like access
    #             t = weights_obj[k]
    #             return to_tensor(t)
    #         except Exception:
    #             pass
    #         try:
    #             # some loaders have get()
    #             if hasattr(weights_obj, "get"):
    #                 t = weights_obj.get(k)
    #                 if t is not None:
    #                     return to_tensor(t)
    #         except Exception:
    #             pass
    #     return None


    # def load_weights(self, weights):
    #     """
    #     Load weights from a vLLM weights container (or a dict-like mapping).
    #     This maps common HF checkpoint keys into the internal submodules.
    #     """

    #     # ensure config rotary/head settings consistent (helps RoPE mismatch)
    #     cfg = getattr(self, "config", None)
    #     if cfg is not None:
    #         head_dim = cfg.hidden_size // cfg.num_attention_heads
    #         # If no explicit rotary_dim, set it to head_dim (common)
    #         if not hasattr(cfg, "rotary_dim") or cfg.rotary_dim is None:
    #             cfg.rotary_dim = head_dim
    #         # some implementations expect rotary_dim <= head_dim
    #         if cfg.rotary_dim > head_dim:
    #             cfg.rotary_dim = head_dim

    #     missing = []
    #     # load embeddings
    #     name = "model.embed_tokens.weight"
    #     t = self._try_get_tensor_from_weights(weights, "embed_tokens.weight") or \
    #         self._try_get_tensor_from_weights(weights, "model.embed_tokens.weight") or \
    #         self._try_get_tensor_from_weights(weights, "transformer.wte.weight")
    #     if t is not None:
    #         try:
    #             param = getattr(self, "model").embed_tokens.weight
    #             param.data.copy_(t.to(param.dtype))
    #         except Exception as e:
    #             missing.append(("embed_tokens", str(e)))
    #     else:
    #         missing.append(("embed_tokens", "not found"))

    #     # load final lm_head / tie embeddings if exists
    #     try:
    #         if hasattr(self, "lm_head") and self.model.lm_head is not None:
    #             t_lm = self._try_get_tensor_from_weights(weights, "lm_head.weight") or self._try_get_tensor_from_weights(weights, "model.lm_head.weight")
    #             if t_lm is not None:
    #                 param = self.model.lm_head.weight
    #                 param.data.copy_(t_lm.to(param.dtype))
    #     except Exception:
    #         pass

    #     try:
    #         if hasattr(self, "rollout_head") and self.model.lm_head is not None:
    #             t_lm = self._try_get_tensor_from_weights(weights, "rollout_head.weight") or self._try_get_tensor_from_weights(weights, "model.rollout_head.weight")
    #             if t_lm is not None:
    #                 param = self.model.rollout_head.weight
    #                 param.data.copy_(t_lm.to(param.dtype))
    #     except Exception:
    #         pass

    #     # Load layerwise weights
    #     n_layers = getattr(cfg, "num_hidden_layers", None)
    #     if n_layers is None:
    #         # fallback try to infer from self.model.layers
    #         try:
    #             n_layers = len(list(self.model.layers))
    #         except Exception:
    #             n_layers = 0

    #     for i in range(n_layers):
    #         prefix = f"model.layers.{i}."
    #         # layer norm before
    #         for ln_name, module_attr in [
    #             ("input_layernorm.weight", "input_layernorm.weight"),
    #             ("post_attention_layernorm.weight", "post_attention_layernorm.weight")
    #         ]:
    #             key = prefix + ln_name
    #             t = self._try_get_tensor_from_weights(weights, key)
    #             if t is None:
    #                 # try without "model."
    #                 t = self._try_get_tensor_from_weights(weights, key[len("model."):])
    #             if t is not None:
    #                 try:
    #                     # module path: self.model.layers[i].input_layernorm.weight
    #                     module = getattr(self.model.layers[i], ln_name.split(".")[0])
    #                     param = getattr(module, "weight")
    #                     param.data.copy_(t.to(param.dtype))
    #                 except Exception as e:
    #                     missing.append((key, str(e)))
    #             else:
    #                 missing.append((key, "not found"))

    #         # ----- Self-attention projections -----
    #         att_prefix = prefix + "self_attn."
    #         # q_proj
    #         q_w = self._try_get_tensor_from_weights(weights, att_prefix + "q_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.q_proj.weight")
    #         q_b = self._try_get_tensor_from_weights(weights, att_prefix + "q_proj.bias") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.q_proj.bias")
    #         if q_w is not None:
    #             try:
    #                 q_param = getattr(self.model.layers[i].self_attn.q_proj, "weight")
    #                 q_param.data.copy_(q_w.to(q_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "q_proj.weight", str(e)))
    #         else:
    #             missing.append((att_prefix + "q_proj.weight", "not found"))

    #         if q_b is not None:
    #             try:
    #                 qb_param = getattr(self.model.layers[i].self_attn.q_proj, "bias")
    #                 qb_param.data.copy_(q_b.to(qb_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "q_proj.bias", str(e)))

    #         # k_proj
    #         k_w = self._try_get_tensor_from_weights(weights, att_prefix + "k_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.k_proj.weight")
    #         k_b = self._try_get_tensor_from_weights(weights, att_prefix + "k_proj.bias") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.k_proj.bias")
    #         if k_w is not None:
    #             try:
    #                 k_param = getattr(self.model.layers[i].self_attn.k_proj, "weight")
    #                 k_param.data.copy_(k_w.to(k_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "k_proj.weight", str(e)))
    #         else:
    #             missing.append((att_prefix + "k_proj.weight", "not found"))
    #         if k_b is not None:
    #             try:
    #                 kb_param = getattr(self.model.layers[i].self_attn.k_proj, "bias")
    #                 kb_param.data.copy_(k_b.to(kb_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "k_proj.bias", str(e)))

    #         # v_proj
    #         v_w = self._try_get_tensor_from_weights(weights, att_prefix + "v_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.v_proj.weight")
    #         v_b = self._try_get_tensor_from_weights(weights, att_prefix + "v_proj.bias") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.v_proj.bias")
    #         if v_w is not None:
    #             try:
    #                 v_param = getattr(self.model.layers[i].self_attn.v_proj, "weight")
    #                 v_param.data.copy_(v_w.to(v_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "v_proj.weight", str(e)))
    #         else:
    #             missing.append((att_prefix + "v_proj.weight", "not found"))
    #         if v_b is not None:
    #             try:
    #                 vb_param = getattr(self.model.layers[i].self_attn.v_proj, "bias")
    #                 vb_param.data.copy_(v_b.to(vb_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "v_proj.bias", str(e)))

    #         # o_proj (output projection)
    #         o_w = self._try_get_tensor_from_weights(weights, att_prefix + "o_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.self_attn.o_proj.weight")
    #         if o_w is not None:
    #             try:
    #                 o_param = getattr(self.model.layers[i].self_attn.o_proj, "weight")
    #                 o_param.data.copy_(o_w.to(o_param.dtype))
    #             except Exception as e:
    #                 missing.append((att_prefix + "o_proj.weight", str(e)))
    #         else:
    #             missing.append((att_prefix + "o_proj.weight", "not found"))

    #         # ----- MLP -----
    #         mlp_prefix = prefix + "mlp."
    #         down_w = self._try_get_tensor_from_weights(weights, mlp_prefix + "down_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.mlp.down_proj.weight")
    #         gate_w = self._try_get_tensor_from_weights(weights, mlp_prefix + "gate_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.mlp.gate_proj.weight")
    #         up_w = self._try_get_tensor_from_weights(weights, mlp_prefix + "up_proj.weight") or self._try_get_tensor_from_weights(weights, f"layers.{i}.mlp.up_proj.weight")
    #         if down_w is not None:
    #             try:
    #                 down_param = getattr(self.model.layers[i].mlp.down_proj, "weight")
    #                 down_param.data.copy_(down_w.to(down_param.dtype))
    #             except Exception as e:
    #                 missing.append((mlp_prefix + "down_proj.weight", str(e)))
    #         else:
    #             missing.append((mlp_prefix + "down_proj.weight", "not found"))
    #         if gate_w is not None:
    #             try:
    #                 gate_param = getattr(self.model.layers[i].mlp.gate_proj, "weight")
    #                 gate_param.data.copy_(gate_w.to(gate_param.dtype))
    #             except Exception as e:
    #                 missing.append((mlp_prefix + "gate_proj.weight", str(e)))
    #         else:
    #             missing.append((mlp_prefix + "gate_proj.weight", "not found"))
    #         if up_w is not None:
    #             try:
    #                 up_param = getattr(self.model.layers[i].mlp.up_proj, "weight")
    #                 up_param.data.copy_(up_w.to(up_param.dtype))
    #             except Exception as e:
    #                 missing.append((mlp_prefix + "up_proj.weight", str(e)))
    #         else:
    #             missing.append((mlp_prefix + "up_proj.weight", "not found"))

    #     # optionally print missing entries summary
    #     if len(missing) > 0:
    #         print(f"[load_weights] Warning: some keys were missing or failed to copy ({len(missing)}). Examples:")
    #         for k, reason in missing[:20]:
    #             print(f"  - {k}: {reason}")
    #     else:
    #         print("[load_weights] All weights loaded (no missing keys detected).")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.use_adaptor:
            logits = lm_logits + self.rollout_head(hidden_states[:, slice_indices, :])
        else:
            logits = lm_logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.lm_logits = lm_logits
        return output


class QwenWithAdaptor(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.adaptor = copy.deepcopy(self.lm_head)
        self.use_adaptor = True

    def reset_adaptor(self):
        self.adaptor = copy.deepcopy(self.lm_head)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        lm_logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.use_adaptor:
            logits = self.adaptor(hidden_states[:, slice_indices, :])
        else:
            logits = lm_logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        output.lm_logits = lm_logits
        return output