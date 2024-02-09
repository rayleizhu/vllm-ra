
export HF_HOME=/mnt/cachenew2/zhulei1/huggingface

# MODEL=$HF_HOME/local/Llama-2-13b-chat-hf
MODEL=$HF_HOME/local/Llama-2-7b-hf
# WORKER=SH-IDC1-10-142-5-17

PROMPT='tests/prompts/mmlu_5shot.txt'
# TEMPLATE='tests/prompts/simple_concat_schema.txt'
TEMPLATE='tests/prompts/llama2_chat_schema.txt'

# srun -p replacement --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --quotatype spot \
#     python -m vllm.entrypoints.api_server \
#         --model $MODEL --swap-space 16 \
#         --disable-log-requests \
#         --enable-relay-attention false \
#         --sys-prompt-file $PROMPT \
#         --sys-schema-file $TEMPLATE


srun -p replacement --gres=gpu:1 --ntasks-per-node=1 --ntasks=1 --quotatype spot \
    python -m vllm.entrypoints.api_server \
        --model $MODEL --swap-space 16 \
        --disable-log-requests \
        --enable-relay-attention false 
