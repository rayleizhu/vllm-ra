from vllm import LLM, SamplingParams


system_prompt = "You are a helpful, respectful and honest assistant created by researchers from ClosedAI. Your nam is ChatPGT. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "

# Sample prompts.
prompts = [
    "Who are you ?",
    "What can you do ?",
    "What's your name ?",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
mode = 'relay'
model = 'TheBloke/Llama-2-7b-Chat-AWQ'
quant = 'awq'
enforce_eager = False

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

if mode == 'concat':
    # Create an LLM.
    llm = LLM(model=model, quantization=quant, enforce_eager=enforce_eager)
    prompts = [ f"{B_INST} {B_SYS}\n{system_prompt.strip()}\n{E_SYS}\n\n{x.strip()} {E_INST}" 
                for x in prompts ]
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
elif mode == 'relay':
    llm = LLM(model=model, quantization=quant, enforce_eager=enforce_eager, enable_relay_attention=True)
    llm.fill_prefix_kv_cache(shared_prefix=f"{B_INST} {B_SYS}\n{system_prompt.strip()}\n{E_SYS}\n\n")
    # TODO: bos token ?
    prompts = [ f"{x.strip()} {E_INST}" for x in prompts]
    outputs = llm.generate(prompts, sampling_params)
else:
    raise ValueError(f'unknown mode {mode}')


# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(output)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    
