import triton
import triton.language as tl

import torch
from torch import Tensor

import math

def relay_fusion(output1:Tensor,
                 lse1:Tensor,
                 output2:Tensor,
                 lse2:Tensor,
                 backend='native') -> Tensor:
    """fusion operation for relay attention
    """
    assert backend in {'native', 'triton'}
    if backend == 'native':
        return _relay_fuse_native(output1, lse1, output2, lse2)
    else:
        return _relay_fuse_triton(output1, lse1, output2, lse2)


def _relay_fuse_native(out_sys:Tensor, lse_sys:Tensor, out_usr:Tensor, lse_usr:Tensor):
    """
    relay fusion with native pytorch operators

    Arguments:
        out_sys: (*, d) tensor, e.g. (bs, len , h, d). 
        lse_sys: (*, 1) tensor.
        out_usr: (*, d) tensor.
        lse_usr: (*, 1) tensor.  
    Return:
        output: fused output
    """
    alpha_sys = 1. / (1. + (lse_usr-lse_sys).exp()) # (bsz, seq_len, head, 1)
    # alpha_sys = alpha_sys.to(out_sys.dtype)
    # out = alpha_sys * out_sys + (1. - alpha_sys) * out_usr # (bsz, seqlen, nhead, hdim)
    out = alpha_sys * out_sys.to(torch.float32) + \
         (1. - alpha_sys) * out_usr.to(torch.float32) # (bsz, seqlen, nhead, hdim)
    return out.to(out_sys.dtype)


def _relay_fuse_triton(out_sys:Tensor, lse_sys:Tensor, out_usr:Tensor, lse_usr:Tensor):
    """
    relay fusion with triton kernel

    Arguments:
        out_sys: (*, d) tensor, e.g. (bs, len , h, d). 
        lse_sys: (*, 1) tensor.
        out_usr: (*, d) tensor.
        lse_usr: (*, 1) tensor.  
    Return:
        output: fused output
    """
    assert out_sys.size() == out_usr.size()
    assert lse_sys.size() == lse_usr.size()
    assert lse_sys.stride(-1) == 1
    assert out_sys.stride(-1) == 1

    out = torch.empty_like(out_sys)
    
    all_heads = math.prod(lse_sys.shape[:-1]) # parallelize over all heads
    n_cols = out_sys.size(-1)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4

    _relay_fuse_kernel[(all_heads,)](
        out_fused_ptr=out,
        out_sys_ptr=out_sys,
        lse_sys_ptr=lse_sys,
        out_usr_ptr=out_usr,
        lse_usr_ptr=lse_usr,
        n_cols=n_cols,
        out_row_stride=n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

@triton.jit
def _relay_fuse_kernel(
        out_fused_ptr, # final output
        out_sys_ptr, lse_sys_ptr,
        out_usr_ptr, lse_usr_ptr,
        n_cols,
        out_row_stride, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0) # parallelize over different heads
    row_start_offset = row_idx * out_row_stride
    lse_sys = tl.load(lse_sys_ptr+row_idx).to(tl.float32)
    lse_usr = tl.load(lse_usr_ptr+row_idx).to(tl.float32)
    rescale_sys = 1. / (1 + tl.exp(lse_usr - lse_sys))
    rescale_usr = 1. - rescale_sys
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    io_mask = col_offsets < n_cols
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    out_sys = tl.load(out_sys_ptr+row_start_offset+col_offsets,
                      mask=io_mask, other=0.)
    out_usr = tl.load(out_usr_ptr+row_start_offset+col_offsets,
                      mask=io_mask, other=0.)
    out_fused = rescale_sys * out_sys + rescale_usr * out_usr
    # save to output tensor
    tl.store(out_fused_ptr+row_start_offset+col_offsets,
             out_fused, mask=io_mask)
