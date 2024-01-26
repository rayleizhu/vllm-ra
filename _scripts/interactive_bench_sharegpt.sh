

MODEL=meta-llama/Llama-2-7b-hf

python -m vllm.entrypoints.api_server \
        --model meta-llama/Llama-2-7b-hf --swap-space 16 \
        --disable-log-requests