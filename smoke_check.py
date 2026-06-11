from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from shared_latent_readout import JsonInstructionDataset


sender = Path("/home/yezhe/伪查询/Qwen3-0.6B")
receiver = Path("/home/yezhe/伪查询/Qwen3-1.7B")
data = Path("/home/yezhe/数据集/swift/OpenHermes-2___5/openhermes2_5.json")

for name, path in [("sender", sender), ("receiver", receiver)]:
    cfg = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    print(name, "hidden", cfg.hidden_size, "layers", cfg.num_hidden_layers, "vocab", len(tok))

ds = JsonInstructionDataset(str(data), max_samples=3)
print("examples", len(ds))
for ex in ds.examples:
    print("sample", len(ex.context), len(ex.question), len(ex.answer))
