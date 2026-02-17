import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
import numpy as np
import os
import config

# --- Configuration ---
MODEL_PATH = "mlx-community/Llama-3.2-1B-Instruct-4bit"
DATA_PATH = "data/train.jsonl"
ADAPTER_PATH = "adapters"
STEPS = 200
BATCH_SIZE = 1 
LR = 1e-4

def main():
    print(">> [Bootstrap] Loading model...")
    model, tokenizer = load(MODEL_PATH)
    
    # 1. Prepare Model for LoRA
    model.freeze()
    
    # FIX: Added 'scale' and 'keys' explicitly
    adapter_config = {
        "keys": ["q_proj", "v_proj"], 
        "rank": 8, 
        "alpha": 16,
        "scale": config.LORA_SCALE,
        "dropout": 0.0
    }
    
    # Apply LoRA conversion
    linear_to_lora_layers(model, num_layers=4, config=adapter_config)
    
    model.train()
    
    # Calculate trainable params
    n_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f">> [Bootstrap] Trainable parameters: {n_params}")
    
    if n_params == 0:
        raise ValueError("[Error] LoRA failed to apply. No trainable parameters found.")

    # 2. Load and Tokenize Data Manually
    print(">> [Bootstrap] Processing data...")
    dataset = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            # Apply chat template
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            # FORCE integer tokenization
            tokens = tokenizer.encode(text)
            dataset.append(mx.array(tokens))
            
    if not dataset:
        raise ValueError("No data found!")

    # 3. Training Loop
    optimizer = optim.AdamW(learning_rate=LR)
    
    def loss_fn(model, X):
        logits = model(X[None, :]) 
        targets = X[None, :]
        
        logits = logits[:, :-1, :]
        targets = targets[:, 1:]
        
        return nn.losses.cross_entropy(logits, targets).mean()

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    print(">> [Bootstrap] Starting Loop...")
    
    losses = []
    for i in range(STEPS):
        idx = np.random.randint(0, len(dataset))
        X = dataset[idx]
        
        if X.shape[0] > 2048:
            X = X[:2048]

        loss, grads = loss_and_grad_fn(model, X)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        losses.append(loss.item())
        
        if (i+1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / 10
            print(f"   Step {i+1}/{STEPS} | Loss: {avg_loss:.4f}")

    # 4. Save Adapters
    print(f">> [Bootstrap] Saving adapters to {ADAPTER_PATH}...")
    os.makedirs(ADAPTER_PATH, exist_ok=True)
    model.save_weights(os.path.join(ADAPTER_PATH, "adapters.safetensors"))
    
    # Save config needed for loading
    with open(os.path.join(ADAPTER_PATH, "adapter_config.json"), "w") as f:
        save_config = adapter_config.copy()
        # Ensure we save the key MLX expects for loading
        if "keys" in save_config:
            save_config["adapter_modules"] = save_config.pop("keys")
        json.dump(save_config, f)
        
    print(">> [Bootstrap] Done. Adversary is ready.")

if __name__ == "__main__":
    main()
