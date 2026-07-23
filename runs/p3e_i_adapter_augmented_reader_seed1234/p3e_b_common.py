import torch

from p3d3_common import MemoryCache
from p3e_a_common import NativeHeadwiseReader, native_memory_to


class SenderNativeHeadwiseCache:
    """Lossless [16,T,1024] -> [16,T,8,128] view over the existing Qwen3-8B cache."""
    def __init__(self, index_path, capacity=4):
        self.inner = MemoryCache(index_path, capacity=capacity)
        self.path, self.root, self.entries = self.inner.path, self.inner.root, self.inner.entries
        self.index = dict(self.inner.index)
        self.index.update({"memory_shape": "[16,T,8,128]", "kv_heads": 8, "head_dim": 128, "lossless_reshape": True})

    def __len__(self): return len(self.inner)

    def load(self, index):
        source = self.inner.load(index); keys, values = source["keys"], source["values"]
        if keys.shape != values.shape or keys.ndim != 3 or keys.shape[-1] != 1024:
            raise RuntimeError(f"Expected flattened Native KV [16,T,1024], got {tuple(keys.shape)}")
        headwise_keys = keys.reshape(keys.shape[0], keys.shape[1], 8, 128)
        headwise_values = values.reshape(values.shape[0], values.shape[1], 8, 128)
        if not torch.equal(headwise_keys.reshape_as(keys), keys) or not torch.equal(headwise_values.reshape_as(values), values):
            raise RuntimeError("Native headwise reshape is not lossless")
        payload = dict(source); payload["keys"], payload["values"] = headwise_keys, headwise_values
        return payload


__all__ = ["SenderNativeHeadwiseCache", "NativeHeadwiseReader", "native_memory_to"]
