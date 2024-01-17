#!/bin/bash

# set -x

# work load config
CONTEXT_LENs=( 64 128 256 ) 
OUPUT_LENs=( 128 256 512 )
# IOs=( "64,128" "128,256" "256,512")
IOs=( "256,512" )
PREFIX_LENs=( 64 128 256 512 1024 )
NUM_REQS=1000

# llama2 30B config
MODEL=meta-llama/Llama-2-7b-hf

NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for IO in ${IOs[@]}; do
    # echo $IO
    IFS=',' read CONTEXT_LEN OUTPUT_LEN <<< "${IO}"
    # echo $CONTEXT_LEN $OUTPUT_LEN
    for PREFIX_LEN in ${PREFIX_LENs[@]}; do
        for BACKEND in vllm vllm+; do
            model_id=$(echo "$MODEL" | tr '/' '.')
            # echo $model_id
            OUTPUT_DIR=outputs/llm_throughput_syn/${model_id}/nreqs_${NUM_REQS}.ctxlen_${CONTEXT_LEN}.outlen_${OUTPUT_LEN}.prefixlen_${PREFIX_LEN}.backend_${BACKEND}
            mkdir -p $OUTPUT_DIR
            python benchmarks/benchmark_throughput.py \
                --num-prompts $NUM_REQS \
                --input-len $CONTEXT_LEN \
                --output-len $OUTPUT_LEN \
                --output-dir $OUTPUT_DIR \
                --prefix-len $PREFIX_LEN \
                --model $MODEL \
                --backend $BACKEND \
                2>&1 | tee -a $OUTPUT_DIR/${NOW}.log
            sleep 1
        done
    done
done
