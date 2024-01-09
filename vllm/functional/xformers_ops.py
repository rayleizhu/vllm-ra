from typing import Optional, Union, Type

import torch
from xformers.ops.fmha.common import Inputs, AttentionBias, AttentionFwOpBase, Context
from xformers.ops.fmha.dispatch import _dispatch_fw, _ensure_op_supports_or_raise
from xformers.ops.fmha import flash

# from xformers.ops.fmha import _memory_efficient_attention_forward, memory_efficient_attention_forward


def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Type[AttentionFwOpBase]] = flash.FwOp,
) -> torch.Tensor:
    """
    Calculates the forward pass of :attr:`xformers.ops.memory_efficient_attention`.
    """
    return _memory_efficient_attention_forward(
        Inputs(
            query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale
        ),
        op=op,
    )

def _memory_efficient_attention_forward(
    inp: Inputs, op: Optional[Type[AttentionFwOpBase]]
) -> torch.Tensor:
    inp.validate_inputs()
    output_shape = inp.normalize_bmhk()
    if op is None:
        op = _dispatch_fw(inp, False)
    else:
        _ensure_op_supports_or_raise(ValueError, "memory_efficient_attention", op, inp)
    
    # FIXME: the dispatched operator may possibly not return lse 
    out, context = op.apply(inp, needs_gradient=False)
    assert isinstance(context, Context)
    lse = context.lse # (bsz, head, num_queries)
    return out.reshape(output_shape), lse
