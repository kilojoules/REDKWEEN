"""Central HuggingFace / PEFT wrapper for Chaos-1B pipeline.

Provides load_model, generate_text, train_lora, and unload_model as drop-in
replacements for the MLX-LM functions used previously.
"""

import gc
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Quantization config (shared across all loads)
# ---------------------------------------------------------------------------
_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model(model_id, adapter_path=None):
    """Load a model (4-bit quantized) and tokenizer, optionally with a PEFT adapter.

    Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.0):
    """Generate text from *prompt*, returning only the newly generated tokens."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    output_ids = model.generate(**inputs, **gen_kwargs)
    # Strip prompt tokens — HF generate returns full sequence
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def train_lora(
    model_id,
    data_path,
    adapter_path,
    num_iters=200,
    batch_size=1,
    lr=1e-4,
    lora_rank=8,
    lora_alpha=16,
    target_modules=None,
):
    """Run a LoRA fine-tune on *model_id* using chat data in *data_path*/train.jsonl.

    Saves PEFT adapter to *adapter_path*.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # --- Load base model in 4-bit ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load training data ---
    train_file = os.path.join(data_path, "train.jsonl")
    dataset = []
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
            dataset.append(tokens)

    if not dataset:
        print("   [Warning] No training data found — skipping training.")
        unload_model(model, tokenizer)
        return

    # --- Training loop ---
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"   Training for {num_iters} steps (dataset size: {len(dataset)})...")

    losses = []
    for step in range(num_iters):
        # Random-sample a mini-batch (bootstrap sampling, matches MLX pattern)
        batch_tokens = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(batch_size)]
        # Pad to same length within the batch
        max_len = min(max(t.shape[0] for t in batch_tokens), 2048)
        input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for j, t in enumerate(batch_tokens):
            t = t[:max_len]
            input_ids[j, :t.shape[0]] = t
            labels[j, :t.shape[0]] = t

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses[-10:]))
            print(f"   Step {step+1}/{num_iters} | Loss: {avg:.4f}")

    # --- Save adapter ---
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"   Adapter saved to {adapter_path}")

    unload_model(model, tokenizer, optimizer)


def load_model_trainable(model_id, adapter_path=None):
    """Load a model with gradients enabled (for GCG attacks).

    Same as load_model but calls prepare_model_for_kbit_training() and
    skips model.eval(), so gradients flow through the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def unload_model(*objects):
    """Delete references and free GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
