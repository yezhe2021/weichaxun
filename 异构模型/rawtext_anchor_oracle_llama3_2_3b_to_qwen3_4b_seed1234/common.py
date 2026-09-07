import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_tensor(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); torch.save(obj, tmp); tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def configuration(mode):
    cfg = read_json(ROOT / 'config.json')
    if mode == 'smoke':
        cfg.update(test_samples=2)
    cfg['mode'] = mode
    cfg['signature'] = digest(cfg)
    return cfg


def run_root(cfg):
    return ROOT / 'runs' / cfg['mode']


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def load_model(cfg, family):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    model = AutoModelForCausalLM.from_pretrained(
        cfg['models'][family], local_files_only=True, dtype=torch.float16,
        attn_implementation=cfg['attention_implementation']).to('cuda').eval().requires_grad_(False)
    expected = {'qwen': (36, 8, 128), 'llama': (28, 8, 128)}[family]
    observed = (model.config.num_hidden_layers, model.config.num_key_value_heads, model.config.head_dim)
    if observed != expected: raise RuntimeError(f'{family} architecture mismatch: {observed}')
    return model


def manifests(cfg, split, labels=False):
    obj = read_json(run_root(cfg) / 'manifests' / f'{split}.json')
    if obj['signature'] != cfg['signature']: raise RuntimeError('Manifest configuration mismatch')
    if labels: return obj['rows']
    return [{k: v for k, v in row.items() if k not in {'gold_index', 'gold_label'}} for row in obj['rows']]
