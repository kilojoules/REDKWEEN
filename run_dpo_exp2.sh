#!/bin/bash
set -e
cd /workspace/red
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# --- Bootstrap 8B adversary (with explicit GPU cleanup) ---
echo "=== Bootstrapping 8B adversary ==="
rm -rf adapters/adapter_model.safetensors 2>/dev/null
python -c "
import gc, torch
from model_utils import train_lora
train_lora(
    model_id='meta-llama/Llama-3.1-8B-Instruct',
    data_path='data',
    adapter_path='adapters',
    num_iters=200,
    batch_size=1,
    lr=1e-4,
)
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f'GPU memory after cleanup: {torch.cuda.memory_allocated()/1e6:.0f} MB')
"
# Wait for GPU memory to fully release
sleep 5
mkdir -p experiments/dpo_8b_v_3b/adapters
cp -r adapters/* experiments/dpo_8b_v_3b/adapters/
echo "   8B bootstrap adapter copied"
echo "   GPU memory check:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# --- DPO Experiment: 8B adv vs 3B victim ---
echo ""
echo "============================================"
echo "=== DPO Experiment: 8B adv vs 3B victim ==="
echo "============================================"
python redkween_loop.py \
  --name dpo_8b_v_3b \
  --adversary-model meta-llama/Llama-3.1-8B-Instruct \
  --victim-model meta-llama/Llama-3.2-3B-Instruct \
  --victim-dpo \
  --rounds 20 \
  --candidates 200

echo "=== DPO 8B vs 3B complete ==="
cat experiments/dpo_8b_v_3b/metrics.jsonl
