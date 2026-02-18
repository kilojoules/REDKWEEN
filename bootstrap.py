from model_utils import train_lora
import config


def main():
    print(">> [Bootstrap] Training initial adversary LoRA...")
    train_lora(
        model_id=config.ADVERSARY_MODEL,
        data_path=config.DATA_PATH,
        adapter_path=config.ADAPTER_PATH,
        num_iters=200,
        batch_size=1,
        lr=1e-4,
    )
    print(">> [Bootstrap] Done. Adversary is ready.")


if __name__ == "__main__":
    main()
