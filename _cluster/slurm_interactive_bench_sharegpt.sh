#!/bin/bash

# set -x

export HF_HOME=/mnt/cachenew2/zhulei1/huggingface

REQ_PER_SECs=( 2.0 4.0 8.0 10.0 12.0 14.0 16.0 )
PREFIX_LENs=( 512 1024 2048 256 128 64 )
MODELs=( $HF_HOME/local/Llama-2-7b-hf $HF_HOME/local/Llama-2-13b-hf )
DATA_JSON=${HF_HOME}/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json

wait_uitil_setup(){
   while true; do
        sleep 5
        # Check if the line contains the desired output
        if grep -q  "INFO:     Application startup complete." $1; then
            echo "server setup done."
            break
        fi
    done
}

run_bench(){
    PREFIX_LEN=$1
    REQ_PER_SEC=$2
    ENABLE_RELAY=true
    MODEL=$4
    DATA_JSON=$5
    PORT=$6
    GPU=A100
    NOW=$( date "+%Y-%m-%d-%H.%M.%S" )
    model_id=$( basename $MODEL )
    OUTPUT_DIR=outputs/interactive_bench_sharegpt/${GPU}/${model_id}/reqrate_${REQ_PER_SEC}-prefixlen_${PREFIX_LEN}-relay_promptcache
    SERVER_LOG=${OUTPUT_DIR}/server_${NOW}.log
    CLIENT_LOG=${OUTPUT_DIR}/client_${NOW}.log
    RESULT_FILE=${OUTPUT_DIR}/benchmark.json
    mkdir -p ${OUTPUT_DIR}
    # Run the server in the background
    python -m vllm.entrypoints.api_server \
            --model $MODEL --swap-space 16 \
            --disable-log-requests \
            --enable-relay-attention $ENABLE_RELAY \
            --pseudo-prefix-len $PREFIX_LEN \
            --port $PORT \
            &> ${SERVER_LOG} &
    wait_uitil_setup ${SERVER_LOG}
    SERVER_PID=$( grep -oP "Started server process \[\K\d+" ${SERVER_LOG} )
    echo $SERVER_PID
    sleep 2
    python benchmarks/benchmark_serving.py \
            --backend vllm \
            --tokenizer $MODEL --dataset ${DATA_JSON} \
            --request-rate $REQ_PER_SEC \
            --result-json $RESULT_FILE \
            --port $PORT \
            2>&1 | tee -a ${CLIENT_LOG}
    sleep 1
    kill -15 $SERVER_PID
    sleep 10
}

export -f wait_uitil_setup
export -f run_bench

port=24595
for MODEL in ${MODELs[@]}; do
    for PREFIX_LEN in ${PREFIX_LENs[@]}; do
        for REQ_PER_SEC in ${REQ_PER_SECs[@]}; do
            port=$((port+1))
            srun -p replacement --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 \
                bash -c "run_bench $PREFIX_LEN $REQ_PER_SEC true $MODEL $DATA_JSON $port" &
        done
    done
done
