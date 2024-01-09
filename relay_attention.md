



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
- [ ] test in eager mode
- [ ] make the implementation work with CUDAGraph
- [ ] modify the paged attention kernel to return log-softmax-exp
- [ ] benchmark with synthetic data
- [ ] benchmark with LongBench

## Trouble shooting

* quantization
    - [vllm-AutoAWQ](https://docs.vllm.ai/en/latest/quantization/auto_awq.html)
    - [Cannot find the config file for awq when load llm with LLaMA-2 + AWQ](https://github.com/vllm-project/vllm/issues/1419)
* model downloading
    - [hf-mirror](https://hf-mirror.com/)

## Referrences

* [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
