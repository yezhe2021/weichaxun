import torch

from translator import NativeKVTranslator, ResidualKVAdapter


if __name__ == "__main__":
    module=ResidualKVAdapter(rank=64)
    module.to("cpu")
    x=torch.randn(1,16,32,8,64); y=torch.randn_like(x)
    fk,fv,dk,dv=module(x,y)
    assert torch.equal(fk,x) and torch.equal(fv,y)
    assert torch.count_nonzero(dk)==0 and torch.count_nonzero(dv)==0
    z=torch.zeros_like(x); fk,fv,_,_=module(z,z)
    assert torch.count_nonzero(fk)==0 and torch.count_nonzero(fv)==0
    assert all(layer.bias is None for layer in module.modules() if isinstance(layer,torch.nn.Linear))
    assert sum(p.numel() for p in module.parameters())==2097152
    translator=NativeKVTranslator("full36_mlp_q4_to_l1")
    source_k=torch.randn(1,36,32,8,128); source_v=torch.randn_like(source_k)
    target_k,target_v=translator(source_k,source_v)
    assert target_k.shape==(1,16,32,8,64) and target_v.shape==target_k.shape
    assert all(layer.bias is None for layer in translator.modules() if isinstance(layer,torch.nn.Linear))
    assert sum(p.numel() for p in translator.parameters())==154140672
    print("Reverse Full36-MLP geometry and Residual64 identity/zero/bias tests passed",flush=True)
