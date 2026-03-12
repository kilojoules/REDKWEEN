#!/bin/bash
set -e
cd /workspace/red
export CUDA_VISIBLE_DEVICES=1

# --- Bootstrap 1B adversary ---
echo "=== Bootstrapping 1B adversary ==="
rm -rf adapters/adapter_model.safetensors 2>/dev/null
python -c "
from model_utils import train_lora
train_lora(
    model_id='meta-llama/Llama-3.2-1B-Instruct',
    data_path='data',
    adapter_path='adapters',
    num_iters=200,
    batch_size=1,
    lr=1e-4,
)
"
mkdir -p experiments/dpo_1b_v_8b/adapters
cp -r adapters/* experiments/dpo_1b_v_8b/adapters/
echo "   1B bootstrap adapter copied to experiments/dpo_1b_v_8b/adapters/"

# --- Bootstrap 8B adversary ---
echo "=== Bootstrapping 8B adversary ==="
rm -rf adapters/adapter_model.safetensors 2>/dev/null
python -c "
from model_utils import train_lora
train_lora(
    model_id='meta-llama/Llama-3.1-8B-Instruct',
    data_path='data',
    adapter_path='adapters',
    num_iters=200,
    batch_size=1,
    lr=1e-4,
)
"
mkdir -p experiments/dpo_8b_v_3b/adapters
cp -r adapters/* experiments/dpo_8b_v_3b/adapters/
echo "   8B bootstrap adapter copied to experiments/dpo_8b_v_3b/adapters/"

# --- DPO Experiment 1: 1B adv vs 8B victim ---
echo ""
echo "============================================"
echo "=== DPO Experiment 1: 1B adv vs 8B victim ==="
echo "============================================"
python redkween_loop.py \
  --name dpo_1b_v_8b \
  --adversary-model meta-llama/Llama-3.2-1B-Instruct \
  --victim-model meta-llama/Llama-3.1-8B-Instruct \
  --victim-dpo \
  --rounds 20 \
  --candidates 200

# --- DPO Experiment 2: 8B adv vs 3B victim ---
echo ""
echo "============================================"
echo "=== DPO Experiment 2: 8B adv vs 3B victim ==="
echo "============================================"
python redkween_loop.py \
  --name dpo_8b_v_3b \
  --adversary-model meta-llama/Llama-3.1-8B-Instruct \
  --victim-model meta-llama/Llama-3.2-3B-Instruct \
  --victim-dpo \
  --rounds 20 \
  --candidates 200

echo ""
echo "=== All DPO experiments complete ==="
echo "=== Results ==="
cat experiments/dpo_1b_v_8b/metrics.jsonl
echo "---"
cat experiments/dpo_8b_v_3b/metrics.jsonl
