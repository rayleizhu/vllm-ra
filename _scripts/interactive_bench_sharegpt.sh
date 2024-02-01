REQ_PER_SECs=( 3.5 )
PREFIX_LENs=( 512 )
MODEL=meta-llama/Llama-2-7b-hf
DATA_JSON=${HF_HOME}/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json

GPU=$( nvidia-smi --query-gpu=name --format=csv | tail -n1 | tr ' ' '-' )
NOW=$( date "+%Y-%m-%d-%H.%M.%S" )

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

for PREFIX_LEN in ${PREFIX_LENs[@]}; do
    for REQ_PER_SEC in ${REQ_PER_SECs[@]}; do
        # model_id=${MODEL#*/}
        model_id=$( basename $MODEL )
        OUTPUT_DIR=outputs/interactive_bench_sharegpt/${GPU}/${model_id}/reqrate_${REQ_PER_SEC}-prefixlen_${PREFIX_LEN}-relay_promptcache
        SERVER_LOG=${OUTPUT_DIR}/server_${NOW}.log
        CLIENT_LOG=${OUTPUT_DIR}/client_${NOW}.log
        RESULT_FILE=${OUTPUT_DIR}/benchmark.json
        mkdir -p ${OUTPUT_DIR}
        # make sure the background server is shutdown
        pkill -f "python -m vllm.entrypoints.api_server"
        # Run the server in the background
        python -m vllm.entrypoints.api_server \
                --model $MODEL --swap-space 16 \
                --disable-log-requests \
                --enable-relay-attention true \
                --pseudo-prefix-le $PREFIX_LEN \
                &> ${SERVER_LOG} &
        wait_uitil_setup ${SERVER_LOG}
        sleep 2
        python benchmarks/benchmark_serving.py \
                --backend vllm \
                --tokenizer $MODEL --dataset ${DATA_JSON} \
                --request-rate $REQ_PER_SEC \
                --result-json $RESULT_FILE \
                2>&1 | tee -a ${CLIENT_LOG}
        sleep 1
        pkill -f "python -m vllm.entrypoints.api_server"
        sleep 1
    done
done

