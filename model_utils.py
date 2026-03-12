"""Central HuggingFace / PEFT wrapper for REDKWEEN pipeline.

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

    # --- Apply LoRA (resume from existing adapter if available) ---
    if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"   [PEFT] Resuming training from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print(f"   [PEFT] Initializing new LoRA adapter (rank={lora_rank})")
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
    dataset = []  # list of (tokens, prompt_len) tuples
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Full conversation tokens
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0)
            # Prompt-only tokens (for label masking — only train on assistant response)
            prompt_messages = row["messages"][:-1]  # everything except assistant reply
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_len = len(tokenizer.encode(prompt_text))
            dataset.append((tokens, prompt_len))

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
        batch_samples = [dataset[random.randint(0, len(dataset) - 1)] for _ in range(batch_size)]
        batch_tokens = [s[0] for s in batch_samples]
        batch_prompt_lens = [s[1] for s in batch_samples]
        # Pad to same length within the batch
        max_len = min(max(t.shape[0] for t in batch_tokens), 2048)
        input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for j, (t, plen) in enumerate(zip(batch_tokens, batch_prompt_lens)):
            t = t[:max_len]
            input_ids[j, :t.shape[0]] = t
            # Only compute loss on assistant response tokens (mask prompt with -100)
            if plen < t.shape[0]:
                labels[j, plen:t.shape[0]] = t[plen:]

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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


def _sequence_logprobs(model, token_ids, prompt_len):
    """Sum of per-token log probs for the response portion (after prompt_len).

    Args:
        model: CausalLM (may be PEFT-wrapped).
        token_ids: 1-D tensor of token IDs for the full sequence.
        prompt_len: number of prompt tokens to skip when computing the sum.

    Returns:
        Scalar tensor (sum of log probs for response tokens).
    """
    ids = token_ids[:2048].unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(ids)
    outputs = model(input_ids=ids, attention_mask=attention_mask)
    logits = outputs.logits  # (1, seq_len, vocab)

    # Shift: logits at position t predict token at t+1
    shift_logits = logits[0, prompt_len - 1:-1, :]
    shift_labels = ids[0, prompt_len:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
    return token_lp.sum()


def train_dpo(
    model_id,
    data_path,
    adapter_path,
    num_iters=200,
    batch_size=1,
    lr=1e-4,
    beta=0.1,
    lora_rank=8,
    lora_alpha=16,
    target_modules=None,
):
    """DPO fine-tune on preference pairs in *data_path*/train_dpo.jsonl.

    Each line: {"prompt": str, "chosen": str, "rejected": str}

    Uses the PEFT disable/enable adapter trick so the reference model and
    the policy share a single set of base weights (no extra VRAM).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

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

    # --- LoRA adapter (resume or init) ---
    if os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        print(f"   [PEFT] Resuming DPO from existing adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        print(f"   [PEFT] Initializing new LoRA adapter for DPO (rank={lora_rank})")
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

    # --- Load preference data ---
    train_file = os.path.join(data_path, "train_dpo.jsonl")
    dataset = []  # list of (chosen_ids, rejected_ids, prompt_len)
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            chosen_msgs = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["chosen"]},
            ]
            rejected_msgs = [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["rejected"]},
            ]
            prompt_msgs = [{"role": "user", "content": row["prompt"]}]

            chosen_text = tokenizer.apply_chat_template(chosen_msgs, tokenize=False)
            rejected_text = tokenizer.apply_chat_template(rejected_msgs, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
            )

            chosen_ids = tokenizer.encode(chosen_text, return_tensors="pt").squeeze(0)
            rejected_ids = tokenizer.encode(rejected_text, return_tensors="pt").squeeze(0)
            prompt_len = len(tokenizer.encode(prompt_text))

            dataset.append((chosen_ids, rejected_ids, prompt_len))

    if not dataset:
        print("   [Warning] No DPO training data — skipping.")
        unload_model(model, tokenizer)
        return

    # --- Training loop ---
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"   DPO training for {num_iters} steps "
          f"(dataset: {len(dataset)} pairs, β={beta})...")

    losses = []
    for step in range(num_iters):
        batch_indices = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]

        total_loss = torch.tensor(0.0, device=model.device)
        for idx in batch_indices:
            chosen_ids, rejected_ids, prompt_len = dataset[idx]

            # Policy log-probs (adapters enabled, gradients flow)
            model.enable_adapter_layers()
            chosen_lp = _sequence_logprobs(model, chosen_ids, prompt_len)
            rejected_lp = _sequence_logprobs(model, rejected_ids, prompt_len)

            # Reference log-probs (adapters disabled, no gradients)
            model.disable_adapter_layers()
            with torch.no_grad():
                ref_chosen_lp = _sequence_logprobs(model, chosen_ids, prompt_len)
                ref_rejected_lp = _sequence_logprobs(model, rejected_ids, prompt_len)
            model.enable_adapter_layers()

            # DPO loss: -log σ(β · (Δchosen - Δrejected))
            chosen_reward = beta * (chosen_lp - ref_chosen_lp)
            rejected_reward = beta * (rejected_lp - ref_rejected_lp)
            loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)
            total_loss = total_loss + loss

        total_loss = total_loss / batch_size
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(total_loss.item())
        if (step + 1) % 10 == 0:
            avg = sum(losses[-10:]) / min(10, len(losses[-10:]))
            print(f"   Step {step+1}/{num_iters} | DPO Loss: {avg:.4f}")

    # --- Save adapter ---
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    print(f"   DPO adapter saved to {adapter_path}")

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


@torch.inference_mode()
def extract_hidden_states(model, tokenizer, texts, layer_idx=None,
                          max_length=512):
    """Extract residual-stream hidden states from *model* at *layer_idx*.

    Runs a forward pass on each text and returns the hidden state at the
    last non-padding token position.  Used by ``sae_analysis.py`` to
    collect activations for sparse autoencoder analysis.

    Args:
        model: HuggingFace CausalLM (may be quantized / PEFT-wrapped).
        tokenizer: corresponding tokenizer.
        texts: list of input strings.
        layer_idx: transformer layer to extract from (0-indexed into
                   ``output_hidden_states``).  ``None`` selects the
                   middle layer (matching the Anthropic SAE setup).
        max_length: tokenizer truncation limit.

    Returns:
        ``torch.Tensor`` of shape ``(len(texts), hidden_dim)`` in float32.
    """
    n_layers = model.config.num_hidden_layers
    if layer_idx is None:
        layer_idx = n_layers // 2

    collected = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length,
        ).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states tuple: [0] = embeddings, [i] = output of layer i-1
        h = outputs.hidden_states[layer_idx]
        seq_len = int(inputs["attention_mask"].sum())
        collected.append(h[0, seq_len - 1, :].float().cpu())

    return torch.stack(collected)


def unload_model(*objects):
    """Delete references and free GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
