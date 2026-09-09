from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from data import cuda


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), -1)


def apply_rope(model, tensor, positions):
    position_ids = torch.tensor([positions], dtype=torch.long, device=tensor.device)
    dummy = torch.empty(1, len(positions), model.config.hidden_size, dtype=tensor.dtype, device=tensor.device)
    cos, sin = model.model.rotary_emb(dummy, position_ids)
    return tensor * cos[0][None, :, None, :] + rotate_half(tensor) * sin[0][None, :, None, :]


def dynamic_cache(model, pre_key, value, positions=None):
    positions = list(range(pre_key.shape[1])) if positions is None else positions
    post_key = apply_rope(model, pre_key, positions)
    items = [(post_key[layer].permute(1,0,2).unsqueeze(0), value[layer].permute(1,0,2).unsqueeze(0)) for layer in range(pre_key.shape[0])]
    return DynamicCache(ddp_cache_data=items, config=model.config)


def answer_logits(model, sample, key=None, value=None, prompt_kind="suffix"):
    prompt = sample["question_suffix_ids"] if prompt_kind == "suffix" else sample["question_only_ids"]
    target = sample["answer_token_ids"]
    current = prompt + target[:-1]
    prefix = 0 if key is None else key.shape[1]
    ids = torch.tensor([current], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix+len(current), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix+len(current), dtype=torch.long, device=cuda())
    kwargs = {}
    if key is not None: kwargs["past_key_values"] = dynamic_cache(model, key, value)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, **kwargs)
    start = len(prompt)-1
    logits = output.logits[0, start:start+len(target)].float()
    gold = torch.tensor(target, dtype=torch.long, device=cuda())
    return logits, gold


def ce_loss(model, sample, key, value):
    logits, gold = answer_logits(model, sample, key, value)
    return F.cross_entropy(logits, gold), logits, gold


def kd_loss(student_logits, teacher_logits, temperature):
    teacher = teacher_logits.to(student_logits.device).float()
    t = float(temperature)
    teacher_prob = (teacher/t).softmax(-1)
    student_log = (student_logits/t).log_softmax(-1)
    teacher_log = (teacher/t).log_softmax(-1)
    return (teacher_prob * (teacher_log-student_log)).sum(-1).mean() * (t*t)


@torch.no_grad()
def generate(model, tok, sample, cfg, key=None, value=None, prompt_kind="suffix"):
    prompt = sample["question_suffix_ids"] if prompt_kind == "suffix" else sample["question_only_ids"]
    prefix = 0 if key is None else key.shape[1]
    ids = torch.tensor([prompt], dtype=torch.long, device=cuda())
    positions = torch.arange(prefix, prefix+len(prompt), device=cuda()).unsqueeze(0)
    mask = torch.ones(1, prefix+len(prompt), dtype=torch.long, device=cuda())
    kwargs = {}
    if key is not None: kwargs["past_key_values"] = dynamic_cache(model, key, value)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True, **kwargs)
    past, next_token = output.past_key_values, output.logits[:,-1].argmax(-1,keepdim=True)
    generated=[]; next_position=prefix+len(prompt)
    for _ in range(cfg["max_new_tokens"]):
        token=int(next_token.item())
        if token == tok.eos_token_id: break
        generated.append(token)
        mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=cuda())],1)
        output=model(input_ids=next_token,attention_mask=mask,position_ids=torch.tensor([[next_position]],device=cuda()),past_key_values=past,use_cache=True)
        past,next_token=output.past_key_values,output.logits[:,-1].argmax(-1,keepdim=True)
        next_position+=1
    return tok.decode(generated,skip_special_tokens=True).strip(), len(generated)


class QueryCapture:
    def __init__(self, model):
        self.values, self.handles = {}, []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook(index), with_kwargs=True))

    def _hook(self, index):
        def hook(module, args, kwargs):
            hidden=kwargs.get("hidden_states",args[0] if args else None)
            shape=(*hidden.shape[:2],-1,module.head_dim)
            query=module.q_norm(module.q_proj(hidden).view(shape))
            self.values[index]=query[0,0].detach().cpu().half()
        return hook

    def close(self):
        for handle in self.handles: handle.remove()


@torch.no_grad()
def capture_first_query(model, sample, key, value, layers):
    capture=QueryCapture(model)
    token=torch.tensor([[sample["question_suffix_ids"][0]]],dtype=torch.long,device=cuda())
    prefix=key.shape[1]
    model(input_ids=token,attention_mask=torch.ones(1,prefix+1,dtype=torch.long,device=cuda()),position_ids=torch.tensor([[prefix]],device=cuda()),past_key_values=dynamic_cache(model,key,value),use_cache=False)
    capture.close()
    if len(capture.values)!=layers: raise RuntimeError("query capture incomplete")
    return torch.stack([capture.values[i] for i in range(layers)])


def functional_diagnostics(model, query_pre, pred_k, pred_v, target_k, target_v, positions, question_position):
    query=query_pre.to(pred_k.device)[:,None]
    query=apply_rope(model,query,[question_position])[:,0]
    pred_k=apply_rope(model,pred_k,positions); target_k=apply_rope(model,target_k,positions)
    pred_k=pred_k.repeat_interleave(4,dim=2); target_k=target_k.repeat_interleave(4,dim=2)
    pred_v=pred_v.repeat_interleave(4,dim=2); target_v=target_v.repeat_interleave(4,dim=2)
    p_logits=torch.einsum("lhd,lthd->lht",query,pred_k)/math.sqrt(pred_k.shape[-1])
    t_logits=torch.einsum("lhd,lthd->lht",query,target_k)/math.sqrt(target_k.shape[-1])
    t_prob=t_logits.float().softmax(-1); p_log=p_logits.float().log_softmax(-1)
    route=(t_prob*(t_prob.clamp_min(1e-9).log()-p_log)).sum(-1).mean(-1)
    p_prob=p_logits.float().softmax(-1)
    t_out=torch.einsum("lht,lthd->lhd",t_prob.to(target_v.dtype),target_v)
    p_out=torch.einsum("lht,lthd->lhd",p_prob.to(pred_v.dtype),pred_v)
    cosine=F.cosine_similarity(t_out.float().flatten(1),p_out.float().flatten(1),1)
    return [{"layer":i,"route_kl":route[i].item(),"attention_output_cosine":cosine[i].item()} for i in range(len(route))]
