import triton
import triton.language as tl

import torch
from torch import Tensor

# import math

def relay_fusion(out_sys:Tensor,
                 lse_sys:Tensor,
                 out_usr:Tensor,
                 lse_usr:Tensor,
                 backend='native',
                 trans_lse_sys:bool=False,
                 trans_lse_usr:bool=False) -> Tensor:
    """fusion operation for relay attention

        Args:
            out_sys, out_usr: shape = [num_tokens, num_heads, head_size]
            lse_sys, lse_usr: shape = [num_tokens, num_heads], or [num_heads, num_tokens] if trans_lse_x=True
            backend: 'native' or 'triton'
            trans_lse_sys, trans_lse_usr: bool flag to specify if lse1 and lse2 should be transposed
        Returns:
            shape = [num_tokens, num_heads, head_size]
    """
    assert backend in {'native', 'triton'}
    assert out_sys.size() == out_usr.size()
    assert out_sys.ndim == 3
    if backend == 'native':
        if trans_lse_sys:
            lse_sys = lse_sys.transpose(0, 1).contiguous()
        if trans_lse_usr:
            lse_usr = lse_usr.transpose(0, 1).contiguous()
        assert lse_sys.size() == out_sys.shape[:2]
        assert lse_usr.size() == out_usr.shape[:2]
        lse_sys = lse_sys.unsqueeze(-1) # (num_tokens, num_heads, 1)
        lse_usr = lse_usr.unsqueeze(-1) # (num_tokens, num_heads, 1)
        alpha_sys = 1. / (1. + (lse_usr-lse_sys).exp()) # (num_tokens, num_heads, 1)
        # alpha_sys = alpha_sys.to(out_sys.dtype)
        # out = alpha_sys * out_sys + (1. - alpha_sys) * out_usr # (num_tokens, nhead, hdim)
        # NOTE (ray) : use fp32 to reduce accumulation error
        out = alpha_sys * out_sys.to(torch.float32) + \
            (1. - alpha_sys) * out_usr.to(torch.float32) # (num_tokens, nhead, hdim)
        out = out.to(out_sys.dtype)
    else:
        out = _relay_fuse_triton(out_sys, lse_sys, out_usr, lse_usr,
                                trans_lse_sys, trans_lse_usr)
    return out


def _relay_fuse_triton(out_sys:Tensor, lse_sys:Tensor, out_usr:Tensor, lse_usr:Tensor,
                       trans_lse_sys:bool, trans_lse_usr:bool):
    # it will be more effeicient to let the final dim contiguous
    assert out_sys.stride(-1) == 1
    assert out_usr.stride(-1) == 1
    out = torch.empty_like(out_sys)
    
    num_tokens, num_heads, head_size = out_sys.size()
    if trans_lse_sys: # (num_heads, num_tokens)
        lse_sys_stride_h, lse_sys_stride_t = lse_sys.stride()
    else:
        lse_sys_stride_t, lse_sys_stride_h = lse_sys.stride()

    if trans_lse_usr: # (num_heads, num_tokens)
        lse_usr_stride_h, lse_usr_stride_t = lse_usr.stride()
    else:
        lse_usr_stride_t, lse_usr_stride_h = lse_usr.stride()

    BLOCK_SIZE = triton.next_power_of_2(head_size)
    # TODO (ray): use tl.autotune to config num_warps
    num_warps = 4
    
    _relay_fuse_kernel[(num_tokens, num_heads)](
        out_fused_ptr=out,
        out_sys_ptr=out_sys,
        lse_sys_ptr=lse_sys,
        out_usr_ptr=out_usr,
        lse_usr_ptr=lse_usr,
        head_size=head_size,
        lse_sys_stride_t=lse_sys_stride_t,
        lse_sys_stride_h=lse_sys_stride_h,
        lse_usr_stride_t=lse_usr_stride_t,
        lse_usr_stride_h=lse_usr_stride_h,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

@triton.jit
def _relay_fuse_kernel(
        out_fused_ptr, # final output
        out_sys_ptr, lse_sys_ptr,
        out_usr_ptr, lse_usr_ptr,
        head_size,
        lse_sys_stride_t, lse_sys_stride_h,
        lse_usr_stride_t, lse_usr_stride_h,
        BLOCK_SIZE: tl.constexpr):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    lse_sys = tl.load(lse_sys_ptr+
                      token_id*lse_sys_stride_t+
                      head_id*lse_sys_stride_h).to(tl.float32)
    lse_usr = tl.load(lse_usr_ptr+
                      token_id*lse_usr_stride_t+
                      head_id*lse_usr_stride_h).to(tl.float32)
    rescale_sys = 1. / (1 + tl.exp(lse_usr - lse_sys))
    rescale_usr = 1. - rescale_sys
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    all_head_id = tl.program_id(0)*tl.num_programs(1) + tl.program_id(1)
    head_offs = tl.arange(0, BLOCK_SIZE)
    io_mask = head_offs < head_size
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    out_sys = tl.load(out_sys_ptr+all_head_id*head_size+head_offs,
                      mask=io_mask, other=0.)
    out_usr = tl.load(out_usr_ptr+all_head_id*head_size+head_offs,
                      mask=io_mask, other=0.)
    out_fused = rescale_sys * out_sys + rescale_usr * out_usr
    # save to output tensor
    tl.store(out_fused_ptr+all_head_id*head_size+head_offs,
             out_fused, mask=io_mask)
