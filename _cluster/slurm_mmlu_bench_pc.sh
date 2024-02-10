

export HF_HOME=/mnt/cachenew2/zhulei1/huggingface
MMLU_ROOT=/mnt/lustrenew/zhulei1/ssd_cache/opencompass/data/mmlu
GPU=A100

NSHOTs=( 5 3 1 )
OUT_LENs=( 64 128 )
ENABLE_RELAYs=( true )
MODELs=( $HF_HOME/local/Llama-2-7b-hf $HF_HOME/local/Llama-2-13b-hf )
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
                srun -p replacement --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --quotatype auto \
                python _scripts/mllu_bench.py \
                    --max-tokens $MAX_TOKENS \
                    --model $MODEL \
                    --mmlu-root $MMLU_ROOT \
                    --nshot $NSHOT \
                    --enable-relay $ENABLE_RELAY \
                    --save-dir $OUTPUT_DIR \
                    --dtype $DTYPE \
                    2>&1 >> $OUTPUT_DIR/${NOW}.log &
                sleep 1
            done
        done
    done
done

