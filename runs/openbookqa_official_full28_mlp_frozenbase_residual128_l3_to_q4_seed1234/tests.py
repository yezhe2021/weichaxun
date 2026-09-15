import torch

from translator import ResidualKVAdapter


if __name__ == "__main__":
    module=ResidualKVAdapter(rank=128)
    module.to("cpu")
    x=torch.randn(1,36,32,8,128); y=torch.randn_like(x)
    fk,fv,dk,dv=module(x,y)
    assert torch.equal(fk,x) and torch.equal(fv,y)
    assert torch.count_nonzero(dk)==0 and torch.count_nonzero(dv)==0
    z=torch.zeros_like(x); fk,fv,_,_=module(z,z)
    assert torch.count_nonzero(fk)==0 and torch.count_nonzero(fv)==0
    assert all(layer.bias is None for layer in module.modules() if isinstance(layer,torch.nn.Linear))
    assert sum(p.numel() for p in module.parameters())==18874368
    print("Residual128 identity/zero/bias/parameter tests passed",flush=True)
