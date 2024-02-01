#!/bin/bash

# set -x
export HF_HOME=/mnt/cachenew2/zhulei1/huggingface

# work load config
IOs=( "64,128" "128,256" "256,512")
# IOs=( "256,512" )
PREFIX_LENs=( 512 1024 2048 )
# PREFIX_LENs=( 1024 )
NUM_REQS=1000
MODELs=( $HF_HOME/local/Llama-2-7b-hf $HF_HOME/local/Llama-2-13b-hf )

GPU=A100
NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for MODEL in ${MODELs[@]}; do
    for IO in ${IOs[@]}; do
        # echo $IO
        IFS=',' read CONTEXT_LEN OUTPUT_LEN <<< "${IO}"
        # echo $CONTEXT_LEN $OUTPUT_LEN
        for PREFIX_LEN in ${PREFIX_LENs[@]}; do
            # model_id=$(echo "$MODEL" | tr '/' '.')
            model_id=$( basename $MODEL )
            echo $model_id
            OUTPUT_DIR=outputs/noninteractive_bench_synthetic/${GPU}/${model_id}/nreqs_${NUM_REQS}.ctxlen_${CONTEXT_LEN}.outlen_${OUTPUT_LEN}.prefixlen_${PREFIX_LEN}.backend_vllm+pc
            mkdir -p $OUTPUT_DIR
            srun -p replacement --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 \
            python benchmarks/benchmark_throughput.py \
                --num-prompts $NUM_REQS \
                --input-len $CONTEXT_LEN \
                --output-len $OUTPUT_LEN \
                --output-dir $OUTPUT_DIR \
                --prefix-len $PREFIX_LEN \
                --model $MODEL \
                --backend vllm+ \
                2>&1 | tee -a $OUTPUT_DIR/${NOW}.log &
            sleep 1
        done
    done
done
