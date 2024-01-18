"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
from pathlib import Path
from typing import Optional
import os.path as osp
import json

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams


def main(args: argparse.Namespace):
    print(args)

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        enable_relay_attention=args.enable_relay
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    if args.prefix_len > 0:
        if args.enable_relay:
            llm.fill_prefix_kv_cache(
                shared_prefix=None,
                shared_prefix_toks=[0]*args.prefix_len)
        else:
            prefix_toks = [0 for _ in range(args.prefix_len)] 
            dummy_prompt_token_ids = [prefix_toks + x 
                for x in dummy_prompt_token_ids]

    def run_to_completion(profile_dir: Optional[str] = None):
        if profile_dir:
            with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        str(profile_dir))) as p:
                llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                             sampling_params=sampling_params,
                             use_tqdm=False)
            print(p.key_averages())
        else:
            start_time = time.perf_counter()
            llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                         sampling_params=sampling_params,
                         use_tqdm=False)
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency

    print("Warming up...")
    run_to_completion(profile_dir=None)

    if args.profile:
        # profile_dir = args.profile_result_dir
        # if not profile_dir:
        #     profile_dir = Path(".") / "vllm_benchmark_result" / f"latency_result_{time.time()}"
        profile_dir = osp.join(args.output_dir, "latency_profile")
        print(f"Profiling (results will be saved to '{profile_dir}')...")
        run_to_completion(profile_dir=profile_dir)
        # NOTE: to avoid any possible overheads, we do not run profiling and 
        # benchmarking in a single process
        return

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile_dir=None))
    avg_latency = np.mean(latencies)
    print(f'Avg latency: {np.mean(avg_latency)} seconds')

    result_file = osp.join(args.output_dir, "latency_benchmark.json")
    with open(result_file, mode='w') as cf:
        json.dump({"avg_latency": avg_latency}, cf, indent=4)


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
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--prefix-len', type=int, default=0)
    parser.add_argument('--enable-relay', default=False, type=str2bool)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', default=False, type=str2bool)
    parser.add_argument('--num-iters',
                        type=int,
                        default=3,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        default=False, type=str2bool,
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        default=False, type=str2bool,
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        '--profile',
        default=False, type=str2bool,
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help=(
            'path to save benchmark outputs. The profile results can be visualized '
            'with ui.perfetto.dev or Tensorboard.'
        ))
    args = parser.parse_args()
    main(args)
