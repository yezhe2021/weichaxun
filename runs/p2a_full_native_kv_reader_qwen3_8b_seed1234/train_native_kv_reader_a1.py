from train_native_kv_reader import main


if __name__ == "__main__":
    main(
        default_reader_rank=32,
        default_gate_init=0.01,
        default_epochs=2,
        default_prefill_final=True,
    )
