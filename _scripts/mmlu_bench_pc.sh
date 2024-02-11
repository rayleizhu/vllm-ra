
MMLU_ROOT=/root/autodl-tmp/opencompass_data/mmlu
GPU=$( nvidia-smi --query-gpu=name --format=csv | tail -n1 | tr ' ' '-' )

NSHOTs=( 5 3 1 )
OUT_LENs=( 32 )
ENABLE_RELAYs=( true )
MODELs=( meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf )
DTYPE='float16'

NOW=$(date "+%Y-%m-%d-%H.%M.%S")

for MAX_TOKENS in ${OUT_LENs[@]}; do
    for MODEL in ${MODELs[@]}; do
        for NSHOT in ${NSHOTs[@]}; do
            for ENABLE_RELAY in ${ENABLE_RELAYs[@]}; do
                model_id=$( basename $MODEL )
                echo $model_id
                OUTPUT_DIR=outputs/mmlu_bench/$GPU/$model_id.$DTYPE/nshot_${NSHOT}.relay_pc.outlen_${MAX_TOKENS}
                mkdir -p $OUTPUT_DIR
                python _scripts/mllu_bench.py \
                    --max-tokens $MAX_TOKENS \
                    --model $MODEL \
                    --mmlu-root $MMLU_ROOT \
                    --nshot $NSHOT \
                    --enable-relay $ENABLE_RELAY \
                    --save-dir $OUTPUT_DIR \
                    --dtype $DTYPE \
                    2>&1 | tee -a $OUTPUT_DIR/${NOW}.log
                sleep 1
            done
        done
    done
done

