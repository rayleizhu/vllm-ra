

# MODEL=meta-llama/Llama-2-7b-hf
# MODEL=microsoft/phi-2
# MODEL=mistralai/Mistral-7B-v0.1
MODEL=mistralai/Mistral-7B-v0.1

export HF_ENDPOINT=https://hf-mirror.com && unset http_proxy && unset https_proxy
huggingface-cli download --resume-download --local-dir-use-symlinks False $MODEL
