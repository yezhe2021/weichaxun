from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import (
    Store, capture_native, cuda, load_json, load_model, prepare_manifest,
    progress, rows_for, save_json, seed_all, tokenizer,
)
from receiver import capture_first_query


def sampled_positions(length, count):
    if length < count: raise RuntimeError(f"context length {length} < sampled tokens {count}")
    values = [(index*(length-1))//(count-1) for index in range(count)]
    if len(set(values)) != count: raise RuntimeError("uniform token sampling produced duplicates")
    return values


@torch.no_grad()
def teacher_logits(model, sample):
    target=sample["answer_token_ids"]
    current=sample["full_input_ids"]+target[:-1]
    ids=torch.tensor([current],dtype=torch.long,device=cuda())
    positions=torch.arange(len(current),device=cuda()).unsqueeze(0)
    output=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=positions,use_cache=False)
    start=len(sample["full_input_ids"])-1
    return output.logits[0,start:start+len(target)].detach().cpu().half()


def build_source(cfg,mode,rows):
    model=load_model(cfg["model_8b"],cfg)
    try:
        for split,samples in rows.items():
            full_manifest=load_json(Path(cfg["work_dir"])/"artifacts"/"manifest.json")[split]
            full_map={sample["id"]:sample for sample in full_manifest}
            required={sample["id"] for sample in samples}|{sample["shuffle_id"] for sample in samples}
            source_samples=[full_map[sample_id] for sample_id in sorted(required)]
            directory=Path(cfg["work_dir"])/"cache"/mode/"source8"/split; directory.mkdir(parents=True,exist_ok=True)
            for index,sample in enumerate(source_samples,1):
                destination=directory/f"{sample['id']}.pt"
                if destination.exists(): continue
                key,value=capture_native(model,sample["context_input_ids"],cfg["num_layers"])
                temporary=destination.with_suffix(".tmp")
                torch.save({"id":sample["id"],"pre_key":key,"value":value,"context_length":len(sample["context_input_ids"])},temporary)
                temporary.replace(destination)
                progress(f"{mode}: source8 {split} {index}/{len(source_samples)}")
    finally:
        del model; torch.cuda.empty_cache()


def build_target_teacher(cfg,mode,rows):
    model=load_model(cfg["model_4b"],cfg)
    try:
        for split,samples in rows.items():
            target_dir=Path(cfg["work_dir"])/"cache"/mode/"target4"/split; target_dir.mkdir(parents=True,exist_ok=True)
            teacher_dir=Path(cfg["work_dir"])/"cache"/mode/"teacher4"/split; teacher_dir.mkdir(parents=True,exist_ok=True)
            for index,sample in enumerate(samples,1):
                target_path=target_dir/f"{sample['id']}.pt"; teacher_path=teacher_dir/f"{sample['id']}.pt"
                if target_path.exists() and teacher_path.exists(): continue
                key,value=capture_native(model,sample["context_input_ids"],cfg["num_layers"])
                positions=sampled_positions(key.shape[1],cfg["sampled_tokens"])
                key_gpu,value_gpu=key.to(cuda()),value.to(cuda())
                query=capture_first_query(model,sample,key_gpu,value_gpu,cfg["num_layers"])
                target={
                    "id":sample["id"],"positions":positions,
                    "pre_key":key[:,positions].contiguous(),"value":value[:,positions].contiguous(),
                    "query_first":query,
                }
                if split=="test":
                    target["full_pre_key"]=key; target["full_value"]=value
                temporary=target_path.with_suffix(".tmp"); torch.save(target,temporary); temporary.replace(target_path)
                logits=teacher_logits(model,sample)
                temporary=teacher_path.with_suffix(".tmp"); torch.save({"id":sample["id"],"logits":logits,"gold":sample["answer_token_ids"]},temporary); temporary.replace(teacher_path)
                del key_gpu,value_gpu
                progress(f"{mode}: target4/teacher {split} {index}/{len(samples)}")
    finally:
        del model; torch.cuda.empty_cache()


def scales(cfg,mode,rows):
    path=Path(cfg["work_dir"])/"artifacts"/mode/"scales.pt"
    if path.exists(): return
    store=Store(cfg,mode,rows)
    sums={name:torch.zeros(cfg["num_layers"],cfg["feature_dim"],dtype=torch.float64) for name in ("source_k","source_v","target_k","target_v")}
    count=0
    for index,sample in enumerate(rows["train"],1):
        source,target=store.source("train",sample["id"]),store.target("train",sample["id"])
        positions=target["positions"]
        values={
            "source_k":source["pre_key"][:,positions].float().flatten(2),
            "source_v":source["value"][:,positions].float().flatten(2),
            "target_k":target["pre_key"].float().flatten(2),
            "target_v":target["value"].float().flatten(2),
        }
        for name,value in values.items(): sums[name]+=value.double().square().sum(1)
        count+=len(positions)
        if index%32==0 or index==len(rows["train"]): progress(f"{mode}: RMS {index}/{len(rows['train'])}")
    output={name:(value/count+1e-6).sqrt().float() for name,value in sums.items()}
    path.parent.mkdir(parents=True,exist_ok=True); torch.save(output,path)
    save_json(path.with_suffix(".json"),{"count_tokens":count,"shape":[cfg["num_layers"],cfg["feature_dim"]],"shared_by_all_writers":True})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--mode",choices=("smoke","development"),required=True); parser.add_argument("action",choices=("manifest","source","target","scales")); args=parser.parse_args()
    cfg=load_json(args.config); seed_all(cfg["seed"])
    if args.action=="manifest": prepare_manifest(cfg); return
    rows=rows_for(cfg,args.mode)
    if args.action=="source": build_source(cfg,args.mode,rows)
    elif args.action=="target": build_target_teacher(cfg,args.mode,rows)
    else: scales(cfg,args.mode,rows)


if __name__=="__main__": main()
