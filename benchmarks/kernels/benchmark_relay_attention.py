import torch
from torch import Tensor
import argparse
import random
import time
import os.path as osp
import json
import sys
import os

from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.input_metadata import InputMetadata

NUM_BLOCKS = 1024
PARTITION_SIZE = 512

@torch.inference_mode()
def main(args):
    rank = int(os.getenv("RANK", "-1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    cuda_visible_device = int(os.getenv('CUDA_VISIBLE_DEVICES'))
    slurm_proc_id = int(os.getenv('SLURM_PROCID'))

    print(dict(rank=rank, local_rank=local_rank, cuda_visible_device=cuda_visible_device, slurm_proc_id=slurm_proc_id))

    assert args.num_query_heads % args.num_kv_heads == 0

    query_dim = args.head_size * args.num_query_heads
    kv_dim = args.head_size * args.num_kv_heads
    scale = args.head_size ** (-0.5)

    attention_fn = PagedAttention(
        num_heads=args.num_query_heads,
        head_size=args.head_size,
        scale=scale,
        num_kv_heads=args.num_kv_heads)

    assert args.phase in ['generation', 'prefill']
    input_len = 1 if args.phase == 'generation' else args.context_len 
    
    # create dummy inputs
    query:Tensor = torch.empty(args.num_reqs, input_len,
        query_dim, dtype=args.dtype, device="cuda")
    query.uniform_(-scale, scale)

    # print(torch.cuda.device_count())
    # print(query.get_device())
    # print(torch.cuda.current_device())

    key:Tensor = torch.empty(args.num_reqs, input_len,
        kv_dim, dtype=args.dtype, device='cuda')
    key.uniform_(-scale, scale)

    value = torch.empty_like(key)
    value.uniform_(-scale, scale)

    # when relay attention is not enabled, just fill this with zero as a placeholder
    prefix_length_buffer = torch.empty(1, dtype=torch.int32, device="cuda")

    # prepare prefix cache buffer
    if args.enable_relay:
        prefix_key_cache = torch.empty(
            1, args.prefix_len, args.num_kv_heads, args.head_size,
            dtype=args.dtype, device="cuda").uniform_(-scale, scale)
        prefix_value_cache = torch.empty_like(
            prefix_key_cache).uniform_(-scale, scale)
        ctx_len = args.context_len
        prefix_len = args.prefix_len
        prefix_length_buffer.fill_(args.prefix_len)
    else:
        prefix_key_cache = None
        prefix_value_cache = None
        ctx_len = args.context_len + args.prefix_len
        prefix_len = 0
        prefix_length_buffer.fill_(0)

    # NOTE: allocate just-enough blocks to profile actual memory usage
    NUM_BLOCKS = args.num_reqs*((ctx_len + args.block_size - 1) // args.block_size)

    if args.phase == 'generation':
        context_lens = [ctx_len for _ in range(args.num_reqs)]
        max_context_len = max(context_lens)
        context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

        # create block tables: (num_reqs, max_num_blocks_per_seq) int tensor
        # the num_reqs*max_num_blocks_per_seq logical blocks will share NUM_BLOCKs phisical blocks
        max_num_blocks_per_seq = (max_context_len + args.block_size - 1) // args.block_size
        block_tables = []
        slot_mapping = []
        for _ in range(args.num_reqs):
            block_table = [
                random.randint(0, NUM_BLOCKS - 1)
                for _ in range(max_num_blocks_per_seq)
            ]
            position = ctx_len - 1
            block_number = block_table[position // args.block_size]
            block_offset = position % args.block_size
            slot = block_number * args.block_size + block_offset # physical slot of the generation token
            slot_mapping.append([slot])
            block_tables.append(block_table)

        block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device="cuda")

        # create the KV cache
        x = 16 // torch.tensor([], dtype=args.dtype).element_size()
        key_cache_shape = (NUM_BLOCKS, args.num_kv_heads,
                        args.head_size // x, args.block_size, x)
        key_cache = torch.empty(size=key_cache_shape, dtype=args.dtype, device="cuda")
        key_cache.uniform_(-scale, scale)
        value_cache_shape = (NUM_BLOCKS, args.num_kv_heads, args.head_size, args.block_size)
        value_cache = torch.empty(size=value_cache_shape,
                                dtype=args.dtype,
                                device="cuda")
        value_cache.uniform_(-scale, scale)
        
        if not args.include_cache_ops:
            slot_mapping = None
        
        input_metadata = InputMetadata(
            prompt_lens=[],
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=args.use_cuda_graph,
            prefix_length=prefix_len,
            prefix_length_buffer=prefix_length_buffer
        )
    else:
        raise NotImplementedError()
    
    # Warm Up. (e.g. compile some JIT kernels)
    print("Warming up...")
    for _ in range(10):
        attention_fn(query, key, value, key_cache, value_cache,
                input_metadata, prefix_key_cache, prefix_value_cache)
    
    # Capture CUDA Graph. 
    # This is useful when batch size is small thus there is CPU bottleneck
    if args.use_cuda_graph:
        print("Capturing CUDA Graph ...")
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            attention_fn(query, key, value, key_cache, value_cache,
                input_metadata, prefix_key_cache, prefix_value_cache)
        attention_fn = lambda *x: g.replay()
        torch.cuda.synchronize()

    def run_profile(save_path):
        torch.profiler._utils._init_for_cuda_graphs()
        prof = torch.profiler.profile()
        torch.cuda.synchronize()
        with prof:
            attention_fn(query, key, value, key_cache, value_cache,
                input_metadata, prefix_key_cache, prefix_value_cache)
        torch.cuda.synchronize()
        prof.export_chrome_trace(save_path)
    
    def run_benchmark(num_iters:int)->float:
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(num_iters):
            attention_fn(query, key, value, key_cache, value_cache,
                input_metadata, prefix_key_cache, prefix_value_cache)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return (end_time - start_time) / num_iters

    # Profile.
    if args.profile:
        print("Run profiling...")
        save_path = osp.join(args.output_dir, "profile.json")
        run_profile(save_path)
    
    # Benchmark.
    print("Run benchmarking...")
    latency = run_benchmark(num_iters=100)

    memory_in_gb = torch.cuda.max_memory_reserved() / 1e9
    latency_in_us = latency * 1e6 

    memory_kv_cache = sys.getsizeof(key_cache.storage()) + \
        sys.getsizeof(value_cache.storage())
    memory_kv_cache /= 1e9
    if prefix_key_cache is not None:
        memory_prefix_kv_cache = sys.getsizeof(prefix_key_cache.storage()) + \
            sys.getsizeof(prefix_value_cache.storage())
    else:
        memory_prefix_kv_cache = 0
    memory_prefix_kv_cache /= 1e9
    
    print(f"Kernel running time: {latency_in_us:.3f} us")
    print(f"Memory used: {memory_in_gb:.02f} GB")
    print(f"KV Cache Memory: {memory_kv_cache:.02f} GB")
    print(f"Prefix KV Cache Memory: {memory_prefix_kv_cache:.02f} GB")

    result_file = osp.join(args.output_dir, "benchmark.json")
    with open(result_file, mode='w') as cf:
        json.dump({"Memory (GB)": memory_in_gb,
                   "Lantency (us)": latency_in_us,
                   "KV mem (GB)": memory_kv_cache,
                   "Prefix KV mem (GB)": memory_prefix_kv_cache
                  },
                cf, indent=4)
    
if __name__ == '__main__':

    def str2bool(v:str):
        """
        Converts string to bool type; enables command line 
        arguments in the format of '--arg1 true --arg2 false'
        """
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    # input config
    parser.add_argument("--num-reqs", type=int, default=64)
    parser.add_argument("--prefix-len", type=int, default=512)
    parser.add_argument("--context-len", type=int, default=128)
    # model config, llama-7b uses 4096 dim = 32 heads * 128 head_size
    # https://huggingface.co/meta-llama/Llama-2-7b/blob/main/params.json
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    # TODO (ray): support alibi
    # parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    
    # TODO (ray): support prefill
    parser.add_argument("--phase", type=str,
                        choices=["generation", 'prefill'],
                        default="generation")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/relay_op/")

    parser.add_argument("--include-cache-ops", default=False, type=str2bool)
    parser.add_argument("--use-cuda-graph", default=False, type=str2bool)
    parser.add_argument("--enable-relay", default=False, type=str2bool)
    parser.add_argument("--profile", default=False, type=str2bool)
    
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    dtype_to_torch_dtype = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
    }
    args.dtype = dtype_to_torch_dtype[args.dtype]

    main(args)
