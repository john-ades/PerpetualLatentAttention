from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from transformers.models.gemma2.modeling_gemma2 import (
    eager_attention_forward,    # for supporting softcap
    logger
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    apply_rotary_pos_emb_interleave,
    DeepseekV3RMSNorm
)


class MLAAttention(nn.Module):
    """
    Modified from `transformers.models.llama.modeling_deepseek_v3.DeepseekV3Attention`
    add support for attention bias and softcapping
    """
    def __init__(self, config, layer_idx: int):

        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.qk_latent_layernorm = getattr(config, "qk_latent_layernorm", True)
        
        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=config.attention_bias)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
            if self.qk_latent_layernorm:
                self.q_a_layernorm = DeepseekV3RMSNorm(self.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=config.attention_bias)

        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        if self.qk_latent_layernorm:
            self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
        )

        self.scaling = self.qk_head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None and "past_key_values" in kwargs:
            past_key_value = kwargs.get("past_key_values")
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        elif self.qk_latent_layernorm:
            q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q_states = self.q_b_proj(self.q_a_proj(hidden_states))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        # ====================================================
        # 1. INJECT M.6 LATENT MEMORY (READ PATH)
        # ====================================================
        memory_latents = kwargs.get("memory_latents", None) 
        
        if memory_latents is not None:
            # FIX: Grab the memory specifically projected for this layer's PCA basis
            if isinstance(memory_latents, (list, tuple)):
                memory_P = memory_latents[self.layer_idx]
            else:
                memory_P = memory_latents
            
            # Ensure batch dimension matches dynamically
            if memory_P.shape[0] != batch_size:
                memory_P = memory_P.expand(batch_size, -1, -1)
                
            S = memory_P.shape[1]
        else:
            memory_P = None
            S = 0

        key_shape = (batch_size, seq_length + S, -1, self.qk_nope_head_dim + self.v_head_dim)

        # ====================================================
        # 2. LET TRANSMLA UNPACK MEMORY NATURALLY
        # ====================================================
        # ✅ CRITICAL FIX: RMSNorm Blowup Prevention
        # Normalize the text tokens FIRST, and then inject the memory slots.
        # This allows memory_P to retain its safe near-zero initialization magnitude!
        if self.qk_latent_layernorm:
            normed_k_pass = self.kv_a_layernorm(k_pass)
        else:
            normed_k_pass = k_pass
            
        if memory_P is not None:
            # ✅ FIX: Normalize memory slots to prevent logit explosion
            if self.qk_latent_layernorm:
                memory_P = self.kv_a_layernorm(memory_P)
            k_pass_combined = torch.cat([memory_P, normed_k_pass], dim=1)
        else:
            k_pass_combined = normed_k_pass
            
        k_pass_proj = self.kv_b_proj(k_pass_combined).view(key_shape).transpose(1, 2)
        
        k_nope, value_states = torch.split(k_pass_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # ====================================================
        # 3. APPLY ROPE STRICTLY TO TEXT TOKENS
        # ====================================================
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # ====================================================
        # 4. ZERO-ROPE PADDING FOR MEMORY
        # ====================================================
        if memory_P is not None:
            # Pad the spatial k_rot dimension with absolute zeros!
            zero_rot = torch.zeros(batch_size, 1, S, self.qk_rope_head_dim, 
                                   device=k_rot.device, dtype=k_rot.dtype)
            k_rot_combined = torch.cat([zero_rot, k_rot], dim=2)
        else:
            k_rot_combined = k_rot

        k_rot_combined = k_rot_combined.expand(*k_nope.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_nope, k_rot_combined), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # ====================================================
        # 4.5 CROSS-DOCUMENT BLOCK MASKING
        # ====================================================
        document_ids = kwargs.get("document_ids", None)
        
        if attention_mask is None:
            past_len = key_states.shape[2] - seq_length - S
            attention_mask = torch.zeros(
                (batch_size, 1, seq_length, seq_length + past_len), 
                device=query_states.device, 
                dtype=query_states.dtype
            )
            if seq_length > 1:
                causal_mask = torch.full(
                    (seq_length, seq_length), 
                    torch.finfo(query_states.dtype).min, 
                    device=query_states.device, 
                    dtype=query_states.dtype
                )
                causal_mask.triu_(diagonal=1)
                attention_mask[..., :, past_len:] = causal_mask

        if document_ids is not None:
            doc_q = document_ids.view(batch_size, 1, seq_length, 1)
            doc_k = document_ids.view(batch_size, 1, 1, seq_length)
            
            # Tokens can only attend if they are in the SAME document AND are not padding
            same_doc = (doc_q == doc_k) & (doc_q != 0)
            
            # Overwrite cross-document connections with -INF
            past_len = key_states.shape[2] - seq_length - S
            attention_mask[..., :, past_len:] = torch.where(
                same_doc, 
                attention_mask[..., :, past_len:], 
                torch.finfo(query_states.dtype).min
            )

        # ====================================================
        # 5. GUARANTEED SAFE-STARTUP GATE MASKING
        # ====================================================
        if memory_P is not None:
            memory_gate = kwargs.get("memory_gate", 0.0)
            if isinstance(memory_gate, torch.Tensor):
                # ✅ FIX: Force exactly 4 dimensions so + memory_gate broadcasts safely
                memory_gate = memory_gate.to(query_states.dtype).view(1, 1, 1, 1)
            else:
                memory_gate = torch.tensor(memory_gate, dtype=query_states.dtype, device=query_states.device).view(1, 1, 1, 1)
                
            # Safely prepend the S memory slots with the learned memory_gate penalty
            memory_mask = attention_mask.new_zeros((batch_size, 1, seq_length, S)) + memory_gate
            attention_mask = torch.cat([memory_mask, attention_mask], dim=-1)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softcap=getattr(self.config, "attn_logit_softcapping", None),
            **kwargs,
        )
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
