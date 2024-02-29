

SRC_HOME=/mnt/cache/share_data/zhangyunchen/llama2_converted
HF_HOME=/mnt/cache/share_data/zhulei1/huggingface

TGT_SRC='meta-llama/Llama-2-7b-hf,llama2-7b'


IFS=',' read TGT SRC <<< "${TGT_SRC}"

MODEL_DIR=$HF_HOME/hub/$(echo "models/$TGT" | sed 's/\//--/g')
mkdir -p $MODEL_DIR

for f in $( ls $SRC_HOME/$SRC ); do
    # echo $f
    ln $SRC_HOME/$SRC/$f $MODEL_DIR/$f
done
