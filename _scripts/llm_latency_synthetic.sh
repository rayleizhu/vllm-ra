
#!/bin/bash

# set -x

# work load config
# IOs=( "64,1" "128,1" "256,1" "64,128" "128,256" "256,512" )
IOs=( "64,1" "128,1" "256,1" )
PREFIX_LENs=( 64 128 256 512 1024 )
RELAY_FLAGs=( false true )
# NUM_REQSs=( 16 64 128 256 )
NUM_REQSs=( 128 256 )

# llama2 7B config
MODEL=meta-llama/Llama-2-7b-hf

NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for NUM_REQS in ${NUM_REQSs[@]}; do
    for IO in ${IOs[@]}; do
        # echo $IO
        IFS=',' read CONTEXT_LEN OUTPUT_LEN <<< "${IO}"
        # echo $CONTEXT_LEN $OUTPUT_LEN
        for PREFIX_LEN in ${PREFIX_LENs[@]}; do
            for RELAY_FLAG in ${RELAY_FLAGs[@]}; do
                model_id=$(echo "$MODEL" | tr '/' '.')
                # echo $model_id
                OUTPUT_DIR=outputs/llm_latency_syn/${model_id}/nreqs_${NUM_REQS}.ctxlen_${CONTEXT_LEN}.outlen_${OUTPUT_LEN}.prefixlen_${PREFIX_LEN}.relay_${RELAY_FLAG}
                mkdir -p $OUTPUT_DIR
                python benchmarks/benchmark_latency.py \
                    --profile false \
                    --model $MODEL \
                    --output-dir $OUTPUT_DIR \
                    --batch-size $NUM_REQS \
                    --input-len $CONTEXT_LEN \
                    --output-len $OUTPUT_LEN \
                    --prefix-len $PREFIX_LEN \
                    --enable-relay $RELAY_FLAG \
                    2>&1 | tee -a $OUTPUT_DIR/${NOW}.log
                sleep 1
            done
        done
    done
done
