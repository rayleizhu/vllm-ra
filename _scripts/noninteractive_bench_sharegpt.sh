data_json="/home/coder/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"
model="TheBloke/Llama-2-7b-Chat-AWQ"
# model="facebook/opt-125m"
prefixlen=256

python benchmarks/benchmark_throughput.py \
    --backend vllm \
    --dataset $data_json \
    --model $model \
    --prefix-len $prefixlen \
    --num-prompts 100