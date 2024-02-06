#!/bin/bash

export HF_HOME=/mnt/cachenew2/zhulei1/huggingface

# PREFIX_LENs=( 512 1024 2048 64 128 256 )
PREFIX_LENs=( 512 1024 2048 )
BACKENDs=( vllm+ vllm )
NUM_REQS=1000
# MODELs=( $HF_HOME/local/Llama-2-7b-hf $HF_HOME/local/Llama-2-13b-hf )
MODELs=( $HF_HOME/local/Llama-2-70b-hf )
TP=2
# model="TheBloke/Llama-2-7b-Chat-AWQ"
# model="facebook/opt-125m"

DATA_JSON=${HF_HOME}/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json
GPU=A100
NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for MODEL in ${MODELs[@]}; do
    for PREFIX_LEN in ${PREFIX_LENs[@]}; do
        # model_id=$(echo "$MODEL" | tr '/' '.')
        model_id=$( basename $MODEL )
        echo $model_id
        # echo $model_id
        OUTPUT_DIR=outputs/noninteractive_bench_sharegpt/${GPU}/${model_id}.tp${TP}/nreqs_${NUM_REQS}.prefixlen_${PREFIX_LEN}.backend_vllm+pc
        mkdir -p $OUTPUT_DIR
        export TOKENIZERS_PARALLELISM=true && \
        srun -p replacement --gres=gpu:8 --ntasks-per-node=1 --ntasks=1 \
        python benchmarks/benchmark_throughput.py \
            --backend vllm+ \
            --dataset $DATA_JSON\
            --model $MODEL \
            --prefix-len $PREFIX_LEN \
            --num-prompts $NUM_REQS \
            --output-dir $OUTPUT_DIR \
            --tensor-parallel-size $TP \
            2>&1 >> $OUTPUT_DIR/${NOW}.log &
        sleep 1
    done
done