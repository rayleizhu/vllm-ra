

srun -p replacement --gres=gpu:8 --ntasks-per-node=1 --ntasks=1 \
    python examples/relay_inference.py