#!/bin/bash

PREFIX_LENs=( 512 1024 2048 )
BACKENDs=( vllm+ )
NUM_REQS=1000
# MODELs=( meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf )
MODELs=( mistralai/Mistral-7B-v0.1 microsoft/phi-2 )
# model="TheBloke/Llama-2-7b-Chat-AWQ"
# model="facebook/opt-125m"

DATA_JSON=${HF_HOME}/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json
GPU=$( nvidia-smi --query-gpu=name --format=csv | tail -n1 | tr ' ' '-' )
NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for MODEL in ${MODELs[@]}; do
    for PREFIX_LEN in ${PREFIX_LENs[@]}; do
        for BACKEND in ${BACKENDs[@]}; do
            # model_id=$(echo "$MODEL" | tr '/' '.')
            model_id=$( basename $MODEL )
            # echo $model_id
            OUTPUT_DIR=outputs/noninteractive_bench_sharegpt/${GPU}/${model_id}/nreqs_${NUM_REQS}.prefixlen_${PREFIX_LEN}.backend_vllm+pc
            # echo $OUTPUT_DIR
            mkdir -p $OUTPUT_DIR
            export TOKENIZERS_PARALLELISM=true && \
            python benchmarks/benchmark_throughput.py \
                --backend $BACKEND \
                --dataset $DATA_JSON\
                --model $MODEL \
                --prefix-len $PREFIX_LEN \
                --num-prompts $NUM_REQS \
                --output-dir $OUTPUT_DIR \
                --load-format dummy \
                2>&1 | tee -a $OUTPUT_DIR/${NOW}.log
            sleep 1
        done
    done
done