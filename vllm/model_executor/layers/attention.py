"""Multi-head attention."""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm._C import ops
from vllm._C import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.utils import is_hip

from vllm.functional.flashattn_ops import flash_attn_forward, flash_attn_with_kvcache
from vllm.functional.relayattn_ops import relay_fusion
from vllm.functional.xformers_ops import memory_efficient_attention_forward

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


class PagedAttention(nn.Module):
    """MHA/MQA/GQA layer with PagedAttention.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Reshape and store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention using either
        xformers or the PagedAttention custom op.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        prefix_key_cache: Optional[torch.Tensor] = None,
        prefix_value_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for the inputs.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        batch_size, seq_len, hidden_size = query.shape

        # fill the prefix kv cache here
        # NOTE: this happens only when we fill prefix kv cache 
        # by passing input_metadata.prefix_length=-1 as the signal
        if input_metadata.prefix_length < 0:
            assert (key_cache is None) and (value_cache is None) and \
                    (prefix_key_cache is not None) and (prefix_value_cache is not None)
            assert input_metadata.is_prompt and (batch_size == 1)
            prefix_key_cache[:, :seq_len, :, :] = key.unflatten(-1, (self.num_kv_heads, self.head_size))
            prefix_value_cache[:, :seq_len, :, :] = value.unflatten(-1, (self.num_kv_heads, self.head_size))
        
        if input_metadata.prefix_length > 0:
            # TODO: check the correctness
            # TODO (ray): use a static tensor to track prefix length to make it compatiable with CUDAGraph
            # FIXME (ray): window attention
            # NOTE: flash attention natively supports MQA/GQA
            assert (prefix_key_cache is not None) and (prefix_value_cache is not None)
            # output_pre, lse_pre = flash_attn_forward(
            #     query.view(1, -1, self.num_heads, self.head_size), # (1, bsz*len, num_heads, head_size)
            #     k=prefix_key_cache, # (1, prefix_length, num_kv_heads, head_size)
            #     v=prefix_value_cache, # (1, prefix_length, num_kv_heads, head_size)
            # )
            # print(input_metadata.prefix_length, input_metadata.prefix_length_buffer)
            output_pre, lse_pre = flash_attn_with_kvcache(
                query.view(1, -1, self.num_heads, self.head_size), # (1, bsz*len, num_heads, head_size)
                k_cache=prefix_key_cache, # (1, prefix_length, num_kv_heads, head_size)
                v_cache=prefix_value_cache, # (1, prefix_length, num_kv_heads, head_size)
                cache_seqlens=input_metadata.prefix_length_buffer, # (1, )
                softmax_scale=self.scale)
            output_pre:torch.Tensor = output_pre.view(-1, self.num_heads, self.head_size)
            # TODO: check if the following caused the silent error with CUDAGraph
            # .view() method does not change the data_ptr(), and keeps launch parameters (stride, size, etc) consistent
            # during capture and replay. However, transpose changes the strides
            # (1, self.num_heads, batch_size*seq_len) -> (batch_size*seq_len, self.num_heads, 1)
            # lse_pre:torch.Tensor = lse_pre.transpose(1, 2).reshape(-1, self.num_heads, 1)
            # lse_pre:torch.Tensor = lse_pre.transpose(1, 2).reshape(-1, self.num_heads, 1).contiguous()
            # lse_pre:torch.Tensor = lse_pre.transpose(0, 2).contiguous()
            # lse_pre:torch.Tensor = lse_pre.squeeze(0).transpose(0, 1).view(-1, self.num_heads, 1).contiguous()
            lse_pre:torch.Tensor = lse_pre.squeeze(0)
            trans_lse_pre = True

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size) # (bsz*seqlen, nheads, head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Reshape the keys and values and store them in the cache.
        # If key_cache and value_cache are not provided, the new key and value
        # vectors will not be cached. This happens during the initial memory
        # profiling run.
        if key_cache is not None and value_cache is not None:
            cache_ops.reshape_and_cache(
                key,
                value,
                key_cache,
                value_cache,
                input_metadata.slot_mapping.flatten(),
            )

        if input_metadata.is_prompt:
            # Prompt run.
            if self.num_kv_heads != self.num_heads:
                # As of Nov 2023, xformers only supports MHA. For MQA/GQA,
                # project the key and value tensors to the desired number of
                # heads.
                # TODO(woosuk): Use MQA/GQA kernels for higher performance.
                query = query.view(query.shape[0], self.num_kv_heads,
                                   self.num_queries_per_kv, query.shape[-1])
                # query: (bsz*seqlen, self.num_kv_heads, self.num_queries_per_kv, head_size)
                # this is for MQA/GQA, see xops.memory_efficient_attention_forward
                # TODO(ray): use native flash attention
                key = key[:, :,
                          None, :].expand(key.shape[0], self.num_kv_heads,
                                          self.num_queries_per_kv,
                                          key.shape[-1])
                value = value[:, :, None, :].expand(value.shape[0],
                                                    self.num_kv_heads,
                                                    self.num_queries_per_kv,
                                                    value.shape[-1])

            # Set attention bias if not provided. This typically happens at the
            # very attention layer of every iteration.
            # FIXME(woosuk): This is a hack.
            if input_metadata.attn_bias is None:
                if self.alibi_slopes is None:
                    attn_bias = BlockDiagonalCausalMask.from_seqlens(
                        [seq_len] * batch_size)
                    if self.sliding_window is not None:
                        attn_bias = attn_bias.make_local_attention(
                            self.sliding_window)
                    input_metadata.attn_bias = attn_bias
                else:
                    input_metadata.attn_bias = _make_alibi_bias(
                        self.alibi_slopes, self.num_kv_heads, batch_size,
                        seq_len, query.dtype)

            # TODO(woosuk): Too many view operations. Let's try to reduce them
            # in the future for code readability.
            if self.alibi_slopes is None:
                # [1, bsz*seqlen, head_groups, num_heads_per_group, K]
                query = query.unsqueeze(0)
                key = key.unsqueeze(0)
                value = value.unsqueeze(0)
            else:
                # [batch, seqlen, head_groups, num_heads_per_group, K]
                query = query.unflatten(0, (batch_size, seq_len))
                key = key.unflatten(0, (batch_size, seq_len))
                value = value.unflatten(0, (batch_size, seq_len))

            # out, lse = xops.memory_efficient_attention_forward(
            #     query,
            #     key,
            #     value,
            #     attn_bias=input_metadata.attn_bias,
            #     p=0.0,
            #     scale=self.scale,
            #     op=xops.fmha.MemoryEfficientAttentionFlashAttentionOp[0] if
            #     (is_hip()) else None,
            # )
            # output = out.view_as(query)
            out, lse = memory_efficient_attention_forward(query,
                key,
                value,
                attn_bias=input_metadata.attn_bias,
                p=0.0,
                scale=self.scale)
            output = out.view(batch_size*seq_len, self.num_heads, self.head_size)
            # (bsz, head, num_queries) -> (bsz*num_queries, nheads, 1)
            lse = lse.transpose(1, 2).reshape(batch_size*seq_len, self.num_heads).contiguous()
        else:
            # Decoding run.
            if key_cache is not None and value_cache is not None:
                # (bsz*seqlen, nheads, head_size)
                # (bsz*seqlen, nheads)
                output, lse = _paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    input_metadata,
                    self.num_kv_heads,
                    self.scale,
                    self.alibi_slopes,
                    use_v1=True, # FIXME(ray): fix this hack
                )
            else:
                # This happens during the initial memory profiling run for
                # CUDA graphs.
                output = torch.zeros_like(query)
                lse = torch.zeros(batch_size*seq_len, self.num_heads)
        
        if input_metadata.prefix_length > 0:
            # print(output.stride())
            # print(output_pre.stride())
            # print(lse.size(), lse.stride())
            # print(lse_pre.size(), lse_pre.stride())
            # print('------')
            output = relay_fusion(output_pre, lse_pre, output, lse,
                                  backend='triton', trans_lse_sys=trans_lse_pre)
            # output = output

        # print(output.stride())
        # Reshape the output tensor.
        # return output.reshape(batch_size, seq_len, hidden_size)
        return output.view(batch_size, seq_len, hidden_size)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
) -> LowerTriangularMaskWithTensorBias:
    bias = torch.arange(seq_len, dtype=dtype, device="cuda")
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(prompt_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    bias = bias[None, :] - bias[:, None]

    # When using custom attention bias, xformers requires the bias to
    # be sliced from a tensor whose length is a multiple of 8.
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    attn_bias = LowerTriangularMaskWithTensorBias(bias)
    return attn_bias


def _paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    input_metadata: InputMetadata,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    use_v1: Optional[bool] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)

    block_size = value_cache.shape[3]
    num_seqs, num_heads, head_size = query.shape
    max_num_partitions = (
        (input_metadata.max_context_len + _PARTITION_SIZE - 1) //
        _PARTITION_SIZE)

    lse = torch.zeros(
            size=(num_seqs, num_heads),
            dtype=torch.float32,
            device=output.device)
    
    # NOTE(woosuk): We use a simple heuristic to decide whether to use
    # PagedAttention V1 or V2. If the number of partitions is 1, we use
    # V1 to avoid the overhead of reduction. Also, if the number of
    # sequences or heads is large, we use V1 since there is enough work
    # to parallelize.
    # TODO(woosuk): Tune this heuristic.
    # For context len > 8192, use V2 kernel to avoid shared memory shortage.
    if use_v1 is None:
        use_v1 = input_metadata.max_context_len <= 8192 and (
            max_num_partitions == 1 or num_seqs * num_heads > 512)
    if use_v1:
        # Run PagedAttention V1.
        ops.paged_attention_v1(
            lse,
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
    else:
        # Run PagedAttention V2.
        assert _PARTITION_SIZE % block_size == 0
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        # exp_sums = torch.empty(
        #     size=(num_seqs, num_heads, max_num_partitions),
        #     dtype=torch.float32,
        #     device=output.device,
        # )
        # max_logits = torch.empty_like(exp_sums)
        # FIXME (ray): make v2 work with CUDAGraph (use static buffer for tmp_lse)
        tmp_lse = torch.empty(
            size=(num_seqs, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        ops.paged_attention_v2(
            lse,
            output,
            tmp_lse,
            tmp_output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            alibi_slopes,
        )
        # TODO(ray): check the case that max_num_partitions > 1
        # max_lse, _ = tmp_lse.max(dim=-1, keepdim=True) # (num_reqs, num_heads, 1)
        # se = (tmp_lse - max_lse).exp().sum(-1) # (num_seqs, num_heads)
        # lse_ = se.log() + max_lse.squeeze(-1)
        # print(lse)
        # print(lse_)
    return output, lse
