
conda create -n vllm python=3.9 -y && conda activate vllm
conda install nvidia/label/cuda-12.1.1::cuda-toolkit 

pip install -v -e .

