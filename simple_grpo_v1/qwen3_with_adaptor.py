from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
import torch
import torch.nn as nn
from typing import Callable, Optional, Union
import copy
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

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


class QwenWithRolloutHead(Qwen3ForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.rollout_head = RolloutHead(config)
        self.use_adaptor = False

    def reset_adaptor(self):
        self.rollout_head.random_init()
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)  # 移动父类所有参数
        if hasattr(self, 'rollout_head'):
            self.rollout_head = self.rollout_head.to(*args, **kwargs)  # 移动 rollout_head
        return self

    @can_return_tuple
    @auto_docstring
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
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
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


class QwenWithAdaptor(Qwen3ForCausalLM):
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