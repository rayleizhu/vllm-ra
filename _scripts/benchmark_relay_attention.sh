#!/bin/bash

set -x

# work load config
NUM_REQS=64
CONTEXT_LEN=128
PREFIX_LENs=( 64 128 256 512 1024 )

# llama2 30B config
NUM_QUERY_HEADS=52
NUM_KV_HEADS=52
HEAD_SIZE=128

NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for PREFIX_LEN in ${PREFIX_LENs[@]}; do
    for ENABLE_RELAY in false true; do
        OUTPUT_DIR=outputs/relay_op/nreqs_${NUM_REQS}.ctxlen_${CONTEXT_LEN}.prefixlen_${PREFIX_LEN}.relay_${ENABLE_RELAY}
        mkdir -p $OUTPUT_DIR
        python benchmarks/kernels/benchmark_relay_attention.py \
            --num-reqs $NUM_REQS \
            --context-len $CONTEXT_LEN \
            --num-query-heads $NUM_QUERY_HEADS \
            --num-kv-heads $NUM_KV_HEADS \
            --head-size $HEAD_SIZE \
            --output-dir $OUTPUT_DIR \
            --profile true \
            --include-cache-ops true \
            --use-cuda-graph true \
            --prefix-len $PREFIX_LEN \
            --enable-relay $ENABLE_RELAY \
            2>&1 | tee -a $OUTPUT_DIR/${NOW}.log
        sleep 1
    done
done
