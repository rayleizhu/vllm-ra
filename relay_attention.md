



## Interface of model forward

* model.forward()
    - [ModelRunner.execute_model()](vllm/worker/model_runner.py)
    - [ModelRunner.capture_model() <- CUDAGraphRunner.capture()](vllm/worker/model_runner.py)
* ModelRunner.execute_model()
    - [ModelRunner.profile_run()]()
    - [ModelRunner.fill_prefix_kv_cache()]()
    - [Worker.execute_model()](vllm/worker/worker.py)
* ModelRunner.profile_run()
    - kv_caches=[(None, None)] * num_layers
    - prefix_kv_caches = [(None, None)] * num_layers
* ModelRunner.fill_prefix_kv_cache()
    - kv_caches = [(None, None)] * num_layers
    - prefix_kv_caches are the allocated buffers if enable_relay_attention
* CUDAGraphRunner.capture()
    - kv_caches are allocated buffers
    - prefix_kv_caches are the allocated buffers if enable_relay_attention
* Worker.execute_model()
    - kv_caches and prefix_kv_caches are both allocated buffers
    - prefix_kv_caches are the allocated buffers if enable_relay_attention 

## Issues

* relay attention may not work properly for window attention

## TODOs

- [x] finish implemenation
- [x] test in eager mode
- make the implementation work with CUDAGraph
    - [x] use a static buffer to track the prefix cache length
    - [x] fix a bug to make paged_attention_v2 work with CUDAGraph
- optimize the implementation further
    - [x] write a relay fusion kernel with triton
    - [x] modify the paged attention kernel to return log-softmax-exp
    - [ ] use native flash attention kernel to support MQA/GQA
- benchmark standalone relay attention (teaser)
    - [x] latency, memory usage, profile
    - [x] cudagraph mode
    - [ ] run benchmark and profiling on A100 and plot figures
- [ ] benchmark with synthetic data
    - [ ] throughput
    - [ ] latency
        - fixed [a bug of vllm](https://github.com/vllm-project/vllm/pull/2398/files/66f1e084c31e09e5225783b3e18659ca5deebaf6) for OPT and LLAMA models
- [ ] benchmark with LongBench
- [ ] check if we need to change the behavior of tokenizer (e.g. avoid prepending bos token)
    - https://huggingface.co/docs/transformers/main_classes/tokenizer

## Usage

1. benchmark throughputs

```bash
python benchmarks/benchmark_throughput.py --backend vllm+ --prefix-len 1024 --input-len 128 --output-len 256
```

2. sample dialogue examples 

```bash
python examples/relay_inference.py
```

## Trouble shooting

* environment setup
    - [conda cudatoolkit](https://anaconda.org/nvidia/cuda-toolkit)
* quantization
    - [vllm-AutoAWQ](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
    - [Cannot find the config file for awq when load llm with LLaMA-2 + AWQ](https://github.com/vllm-project/vllm/issues/1419)
* model downloading
    - [hf-mirror](https://hf-mirror.com/)
* relay attention does not work with CUDAGraph
    - [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
    - [https://github.com/pytorch/pytorch/issues/114048](https://github.com/pytorch/pytorch/issues/114048)
    - [CUDA semantics - CUDA Graphs](https://pytorch.org/docs/master/notes/cuda.html#constraints)
    - [Transposed read/writes](https://github.com/openai/triton/issues/176)
    - [Visualize DOT graph online for debugging CUDAGraph](https://edotor.net/)
    