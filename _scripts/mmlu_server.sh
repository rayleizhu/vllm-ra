
MODEL=TheBloke/Llama-2-7b-Chat-AWQ
QUANT='awq'

# MODEL='meta-llama/Llama-2-7b-hf'
PROMPT='tests/prompts/mmlu_5shot.txt'
# TEMPLATE='tests/prompts/simple_concat_schema.txt'
TEMPLATE='tests/prompts/llama2_chat_schema.txt'

python -m vllm.entrypoints.api_server \
    --model $MODEL --swap-space 16 \
    --disable-log-requests \
    --enable-relay-attention false \
    --sys-prompt-file $PROMPT \
    --sys-schema-file $TEMPLATE \
    -q $QUANT
