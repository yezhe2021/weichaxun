from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from data import (
    Store, cuda, load_json, load_model, normalize_answer, progress,
    representation_loss, rows_for, save_json, seed_all, token_f1, tokenizer,
)
from receiver import ce_loss, functional_diagnostics, generate, kd_loss
from writers import make_writer, parameter_report, rms_only


def scales_for(cfg,mode):
    return torch.load(Path(cfg["work_dir"])/"artifacts"/mode/"scales.pt",map_location="cpu",weights_only=False)


def writer_for(cfg,mode,kind,checkpoint=None):
    writer=make_writer(kind,scales_for(cfg,mode),cfg).to(cuda())
    if checkpoint:
        state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        writer.load_state_dict(state["writer"] if "writer" in state else state)
    return writer


def save_writer(path,writer,extra=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    payload={"writer":{k:v.detach().cpu() for k,v in writer.state_dict().items()}}
    if extra: payload.update(extra)
    torch.save(payload,path)


def sample_rep(cfg,writer,store,split,sample):
    source,target=store.source(split,sample["id"]),store.target(split,sample["id"])
    positions=target["positions"]
    sk=source["pre_key"][:,positions].to(cuda()); sv=source["value"][:,positions].to(cuda())
    tk=target["pre_key"].to(cuda()); tv=target["value"].to(cuda())
    pk,pv=writer(sk,sv)
    return (*representation_loss(pk,pv,tk,tv,cfg["stage_a_cosine_weight"]),pk,pv,tk,tv,target)


@torch.no_grad()
def rep_validation(cfg,writer,store,split,samples,diagnostic_model=None):
    writer.eval(); losses=[]; all_layers=[]; diagnostic=None
    for index,sample in enumerate(samples):
        loss,layers,pk,pv,tk,tv,target=sample_rep(cfg,writer,store,split,sample)
        losses.append(loss.item()); all_layers.append(layers)
        if index==0 and diagnostic_model is not None:
            diagnostic=functional_diagnostics(
                diagnostic_model,target["query_first"],pk,pv,tk,tv,
                target["positions"],sample["context_length"],
            )
    result=[]
    for layer in range(cfg["num_layers"]):
        result.append({"layer":layer,**{key:sum(x[layer][key] for x in all_layers)/len(all_layers) for key in ("k_nmse","v_nmse","k_cosine","v_cosine")}})
        if diagnostic: result[-1].update({"route_kl":diagnostic[layer]["route_kl"],"attention_output_cosine":diagnostic[layer]["attention_output_cosine"]})
    writer.train(); return sum(losses)/len(losses),result


def train_stage_a(cfg,mode,kind,overfit=False):
    rows=rows_for(cfg,mode); store=Store(cfg,mode,rows); writer=writer_for(cfg,mode,kind).train()
    stage_name=f"{kind}_{'overfit16' if overfit else 'stage_a'}"; out=Path(cfg["work_dir"])/"artifacts"/mode/stage_name; out.mkdir(parents=True,exist_ok=True)
    maximum=cfg["smoke_updates"] if mode=="smoke" else (cfg["overfit_updates"] if overfit else cfg["stage_a_updates"])
    # An overfit run can finish all requested updates and then fail only its
    # diagnostic gate.  Re-evaluate that completed artifact under the current,
    # explicit gate instead of throwing away the expensive GPU work.
    summary_path=out/"summary.json"; history_path=out/"history.json"; best_path=out/"best.pt"
    if overfit and mode!="smoke" and summary_path.exists() and history_path.exists() and best_path.exists():
        previous=load_json(summary_path); previous_history=load_json(history_path)
        complete=bool(previous_history) and int(previous_history[-1].get("update",0))>=maximum
        ratio=float(previous.get("loss_ratio",float("inf")))
        missing=list(previous.get("missing_gradient_parameters",[]))
        passed=complete and not missing and torch.isfinite(torch.tensor(ratio)).item() and ratio<=cfg["overfit_required_ratio"]
        if complete:
            previous.update({
                "passed":passed,
                "gate_required_ratio":cfg["overfit_required_ratio"],
                "gate_minimum_relative_improvement":1.0-cfg["overfit_required_ratio"],
                "gate_revalidated_from_completed_run":True,
            })
            save_json(summary_path,previous)
            progress(f"{mode}: {stage_name} reuse completed {maximum}-update run ratio={ratio:.6f} passed={passed}")
            if passed or not cfg["enforce_overfit_gate"]: return
            raise RuntimeError(f"{kind} overfit gate failed")
    grad_acc=cfg["smoke_gradient_accumulation"] if mode=="smoke" else cfg["gradient_accumulation"]
    samples=list(rows["train"][:min(cfg["overfit_samples"],len(rows["train"]))] if overfit else rows["train"])
    validation=samples if overfit else rows["validation"]; validation_split="train" if overfit else "validation"
    optimizer=AdamW(writer.parameters(),lr=cfg["stage_a_lr"],weight_decay=cfg["weight_decay"])
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); initial,_=rep_validation(cfg,writer,store,validation_split,validation)
    history=[]; evaluations=[]; best=float("inf"); cursor=epoch=0; ever_grad={name:False for name,p in writer.named_parameters() if p.requires_grad}
    for update in range(1,maximum+1):
        optimizer.zero_grad(set_to_none=True); batch_losses=[]
        for _ in range(grad_acc):
            if cursor==0: random.Random(cfg["seed"]+epoch).shuffle(samples); epoch+=1
            sample=samples[cursor]; cursor=(cursor+1)%len(samples)
            loss,*_=sample_rep(cfg,writer,store,"train",sample); (loss/grad_acc).backward(); batch_losses.append(loss.item())
        for parameter_name,p in writer.named_parameters():
            if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max().item()>0: ever_grad[parameter_name]=True
        norm=torch.nn.utils.clip_grad_norm_(writer.parameters(),cfg["gradient_clip"]).item(); optimizer.step()
        if not torch.isfinite(torch.tensor(batch_losses)).all(): raise RuntimeError("non-finite Stage A loss")
        history.append({"update":update,"loss":sum(batch_losses)/len(batch_losses),"gradient_norm":norm})
        interval=1 if mode=="smoke" else cfg["stage_a_eval_every"]
        if update%interval==0 or update==maximum:
            value,layers=rep_validation(cfg,writer,store,validation_split,validation)
            selected=value<best
            evaluations.append({"update":update,"validation_loss":value,"selected":selected,"layers":layers})
            if selected: best=value; save_writer(out/"best.pt",writer,{"update":update,"validation_loss":value})
            progress(f"{mode}: {stage_name} {update}/{maximum} validation={value:.6f}")
    diagnostic_model=None
    if not overfit:
        diagnostic_model=load_model(cfg["model_4b"],cfg)
    final,layers=rep_validation(cfg,writer,store,validation_split,validation,diagnostic_model)
    if diagnostic_model is not None:
        del diagnostic_model; torch.cuda.empty_cache()
    elapsed=time.perf_counter()-start; missing=[name for name,value in ever_grad.items() if not value]
    ratio=final/max(initial,1e-12); passed=(not missing and ratio<=cfg["overfit_required_ratio"]) if overfit else True
    if mode=="smoke": passed=not missing
    summary={"kind":kind,"overfit":overfit,"initial_loss":initial,"final_loss":final,"loss_ratio":ratio,"passed":passed,"missing_gradient_parameters":missing,"parameters":parameter_report(writer),"training_seconds":elapsed,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"best_validation_loss":best}
    if overfit:
        summary.update({"gate_required_ratio":cfg["overfit_required_ratio"],"gate_minimum_relative_improvement":1.0-cfg["overfit_required_ratio"]})
    if kind=="full": summary["writer_diagnostics"]=writer.diagnostics()
    save_json(out/"history.json",history); save_json(out/"evaluations.json",evaluations); save_json(out/"final_layer_metrics.json",layers); save_json(out/"summary.json",summary)
    if overfit and cfg["enforce_overfit_gate"] and not passed and mode!="smoke": raise RuntimeError(f"{kind} overfit gate failed")


def full_source(cfg,writer,store,split,sample,source_id=None):
    record=store.source(split,source_id or sample["id"])
    return writer(record["pre_key"].to(cuda()),record["value"].to(cuda()))


@torch.no_grad()
def validation_nll(cfg,model,writer,store,samples):
    writer.eval(); values=[]
    for sample in samples:
        key,value=full_source(cfg,writer,store,"validation",sample)
        loss,_,_=ce_loss(model,sample,key,value); values.append(loss.item())
    writer.train(); return sum(values)/len(values)


@torch.no_grad()
def validation_generation(cfg,model,tok,writer,store,samples):
    writer.eval(); scores=[]
    for sample in samples:
        key,value=full_source(cfg,writer,store,"validation",sample)
        prediction,_=generate(model,tok,sample,cfg,key,value)
        scores.append(token_f1(prediction,sample["answer"]))
    writer.train(); return sum(scores)/len(scores)


def stage_b_name(kind,branch): return f"{kind}_{branch}"


def train_stage_b(cfg,mode,kind,branch):
    rows=rows_for(cfg,mode); store=Store(cfg,mode,rows)
    initial=None if branch=="f1_ce" else Path(cfg["work_dir"])/"artifacts"/mode/f"{kind}_stage_a"/"best.pt"
    writer=writer_for(cfg,mode,kind,initial).train(); model=load_model(cfg["model_4b"],cfg); tok=tokenizer(cfg["model_4b"])
    out=Path(cfg["work_dir"])/"artifacts"/mode/stage_b_name(kind,branch); out.mkdir(parents=True,exist_ok=True)
    formal_max=cfg["direct_ce_updates"] if branch=="f1_ce" else cfg["stage_b_updates"]
    maximum=cfg["smoke_updates"] if mode=="smoke" else formal_max
    grad_acc=cfg["smoke_gradient_accumulation"] if mode=="smoke" else cfg["gradient_accumulation"]
    optimizer=AdamW(writer.parameters(),lr=cfg["stage_b_lr"],weight_decay=cfg["weight_decay"])
    samples=list(rows["train"]); cursor=epoch=0; history=[]; evaluations=[]; best_nll=float("inf"); best_f1=-1.0
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    if mode!="smoke" and 0 in cfg["checkpoint_updates"]:
        save_writer(out/"update_000.pt",writer,{"update":0})
    for update in range(1,maximum+1):
        optimizer.zero_grad(set_to_none=True); values=[]
        for _ in range(grad_acc):
            if cursor==0: random.Random(cfg["seed"]+epoch).shuffle(samples); epoch+=1
            sample=samples[cursor]; cursor=(cursor+1)%len(samples)
            key,value=full_source(cfg,writer,store,"train",sample)
            ce,logits,_=ce_loss(model,sample,key,value); loss=ce
            kd_value=0.0
            if branch=="f2_kd":
                teacher=store.teacher("train",sample["id"])["logits"]
                kd=kd_loss(logits,teacher,cfg["kd_temperature"]); loss=ce+cfg["kd_lambda"]*kd; kd_value=kd.item()
            (loss/grad_acc).backward(); values.append({"loss":loss.item(),"ce":ce.item(),"kd":kd_value})
        norm=torch.nn.utils.clip_grad_norm_(writer.parameters(),cfg["gradient_clip"]).item(); optimizer.step()
        history.append({"update":update,"loss":sum(x["loss"] for x in values)/len(values),"ce":sum(x["ce"] for x in values)/len(values),"kd":sum(x["kd"] for x in values)/len(values),"gradient_norm":norm})
        nll_interval=1 if mode=="smoke" else cfg["nll_eval_every"]
        gen_interval=1 if mode=="smoke" else cfg["generation_eval_every"]
        row={"update":update}
        if update%nll_interval==0 or update==maximum:
            value=validation_nll(cfg,model,writer,store,rows["validation"]); row["validation_nll"]=value
            if value<best_nll: best_nll=value; save_writer(out/"best_nll.pt",writer,{"update":update,"validation_nll":value})
        if update%gen_interval==0 or update==maximum:
            count=cfg["smoke_generation_eval_samples"] if mode=="smoke" else cfg["generation_eval_samples"]
            value=validation_generation(cfg,model,tok,writer,store,rows["validation"][:count]); row["validation_generation_f1"]=value
            if value>best_f1: best_f1=value; save_writer(out/"best_generation.pt",writer,{"update":update,"validation_generation_f1":value})
        if len(row)>1: evaluations.append(row); progress(f"{mode}: {kind}/{branch} {update}/{maximum}")
        if mode!="smoke" and update in cfg["checkpoint_updates"]: save_writer(out/f"update_{update:03d}.pt",writer,{"update":update})
    save_json(out/"history.json",history); save_json(out/"evaluations.json",evaluations)
    save_json(out/"summary.json",{"kind":kind,"branch":branch,"initialized_from":str(initial) if initial else "structure_identity","best_validation_nll":best_nll,"best_validation_generation_f1":best_f1,"training_seconds":time.perf_counter()-start,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"parameters":parameter_report(writer),"stage_b_representation_tether":False})
    del model; torch.cuda.empty_cache()


def aggregate_generation(rows):
    result={}
    for condition in sorted({x["condition"] for x in rows}):
        values=[x for x in rows if x["condition"]==condition]
        result[condition]={"count":len(values),"em":sum(x["em"] for x in values)/len(values),"f1":sum(x["f1"] for x in values)/len(values),"nll":sum(x["nll"] for x in values)/len(values),"teacher_kl":sum(x["teacher_kl"] for x in values)/len(values),"bridge_f1":sum(x["f1"] for x in values if x["type"]=="bridge")/max(sum(x["type"]=="bridge" for x in values),1),"comparison_f1":sum(x["f1"] for x in values if x["type"]=="comparison")/max(sum(x["type"]=="comparison" for x in values),1),"latency_seconds":sum(x["latency_seconds"] for x in values)/len(values)}
    return result


@torch.no_grad()
def evaluate_condition(cfg,model,tok,store,samples,condition,writer=None,kind="correct",raw_kind=None):
    rows=[]
    if writer: writer.eval()
    scales=scales_for(cfg,store.mode)
    for sample in samples:
        source_id=sample["shuffle_id"] if kind=="shuffled" else sample["id"]
        record=store.source("test",source_id); key,value=record["pre_key"].to(cuda()),record["value"].to(cuda())
        started=time.perf_counter()
        if writer: key,value=writer(key,value)
        elif raw_kind=="rms": key,value=rms_only(key,value,scales)
        loss,logits,gold=ce_loss(model,sample,key,value)
        prediction,length=generate(model,tok,sample,cfg,key,value)
        latency=time.perf_counter()-started
        teacher=store.teacher("test",sample["id"])["logits"].to(logits.device).float()
        kl=(teacher.softmax(-1)*(teacher.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).mean().item()
        rows.append({"sample_id":sample["id"],"type":sample["type"],"condition":condition,"answer":sample["answer"],"prediction":prediction,"em":float(normalize_answer(prediction)==normalize_answer(sample["answer"])),"f1":token_f1(prediction,sample["answer"]),"nll":loss.item(),"teacher_kl":kl,"output_tokens":length,"latency_seconds":latency})
    return rows


def baseline(cfg,mode):
    rows=rows_for(cfg,mode); store=Store(cfg,mode,rows); model=load_model(cfg["model_4b"],cfg); tok=tokenizer(cfg["model_4b"])
    output=[]
    output+=evaluate_condition(cfg,model,tok,store,rows["test"],"raw_8b",raw_kind="raw")
    output+=evaluate_condition(cfg,model,tok,store,rows["test"],"rms_only_8b",raw_kind="rms")
    out=Path(cfg["work_dir"])/"artifacts"/mode/"baseline"; out.mkdir(parents=True,exist_ok=True)
    save_json(out/"per_sample.json",output); save_json(out/"summary.json",aggregate_generation(output)); del model; torch.cuda.empty_cache()


def f0(cfg,mode):
    rows=rows_for(cfg,mode); store=Store(cfg,mode,rows); model=load_model(cfg["model_4b"],cfg); tok=tokenizer(cfg["model_4b"]); output=[]
    for kind in ("linear","full"):
        checkpoint=Path(cfg["work_dir"])/"artifacts"/mode/f"{kind}_stage_a"/"best.pt"
        writer=writer_for(cfg,mode,kind,checkpoint); output+=evaluate_condition(cfg,model,tok,store,rows["test"],f"{kind}_f0",writer)
        del writer; torch.cuda.empty_cache()
    out=Path(cfg["work_dir"])/"artifacts"/mode/"f0"; out.mkdir(parents=True,exist_ok=True); save_json(out/"per_sample.json",output); save_json(out/"summary.json",aggregate_generation(output)); del model; torch.cuda.empty_cache()


def final_evaluate(cfg,mode):
    rows=rows_for(cfg,mode); store=Store(cfg,mode,rows); model=load_model(cfg["model_4b"],cfg); tok=tokenizer(cfg["model_4b"]); output=[]
    for stage in ("baseline","f0"):
        output+=load_json(Path(cfg["work_dir"])/"artifacts"/mode/stage/"per_sample.json")
    candidates={"linear":[],"full":[]}
    for kind in ("linear","full"):
        for branch in ("f1_ce","f2_ce","f2_kd"):
            root=Path(cfg["work_dir"])/"artifacts"/mode/stage_b_name(kind,branch)
            summary=load_json(root/"summary.json")
            for checkpoint_name in ("best_nll","best_generation"):
                writer=writer_for(cfg,mode,kind,root/f"{checkpoint_name}.pt")
                condition=f"{kind}_{branch}_{checkpoint_name}"
                output+=evaluate_condition(cfg,model,tok,store,rows["test"],condition,writer)
                if checkpoint_name=="best_generation":
                    candidates[kind].append((summary["best_validation_generation_f1"],condition,root/f"{checkpoint_name}.pt"))
                del writer; torch.cuda.empty_cache()
        _,best_condition,best_path=max(candidates[kind],key=lambda x:x[0])
        writer=writer_for(cfg,mode,kind,best_path)
        output+=evaluate_condition(cfg,model,tok,store,rows["test"],f"best_{kind}_shuffled",writer,"shuffled")
        del writer; torch.cuda.empty_cache()
    summary=aggregate_generation(output)
    reference=load_json(Path(cfg["audit_4b_dir"])/"artifacts"/"development"/"summary.json")["conditions"]
    for name in ("question_only","full_context_text","official_native_cache","shuffled_native_cache"):
        summary[name]=reference[name]
    floor,upper=reference["question_only"]["f1"],reference["full_context_text"]["f1"]
    for name,value in summary.items():
        if "f1" in value: value["recovery"]=(value["f1"]-floor)/max(upper-floor,1e-12)
    best_linear=max((v["f1"],k) for k,v in summary.items() if k.startswith("linear_") and "shuffled" not in k)
    best_full=max((v["f1"],k) for k,v in summary.items() if k.startswith("full_") and "shuffled" not in k)
    conclusion={
        "best_linear":best_linear[1],"best_full":best_full[1],
        "structure_f1_gain":best_full[0]-best_linear[0],
        "linear_correct_minus_shuffled":best_linear[0]-summary["best_linear_shuffled"]["f1"],
        "full_correct_minus_shuffled":best_full[0]-summary["best_full_shuffled"]["f1"],
        "linear_correct_minus_no_memory":best_linear[0]-floor,
        "full_correct_minus_no_memory":best_full[0]-floor,
        "writer_output_cache_bytes_fp16_mean":sum(x["context_length"] for x in rows["test"])/len(rows["test"])*cfg["num_layers"]*cfg["feature_dim"]*2*2,
        "full_writer_ablation_authorized":best_full[0]>best_linear[0],
    }
    out=Path(cfg["work_dir"])/"artifacts"/mode/"evaluation"; out.mkdir(parents=True,exist_ok=True); save_json(out/"per_sample.json",output); save_json(out/"summary.json",summary); save_json(out/"conclusion.json",conclusion); save_json(out/"completion.json",{"completed":True,"training_performed":True,"reader":"identity","joint_reader_training":False})
    del model; torch.cuda.empty_cache(); progress(f"{mode}: final evaluation completed")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--mode",choices=("smoke","development"),required=True); parser.add_argument("action"); args=parser.parse_args(); cfg=load_json(args.config); seed_all(cfg["seed"])
    actions={
        "baseline":baseline,
        "overfit_linear":lambda c,m:train_stage_a(c,m,"linear",True),"overfit_full":lambda c,m:train_stage_a(c,m,"full",True),
        "stagea_linear":lambda c,m:train_stage_a(c,m,"linear",False),"stagea_full":lambda c,m:train_stage_a(c,m,"full",False),"f0":f0,
        "linear_f1":lambda c,m:train_stage_b(c,m,"linear","f1_ce"),"full_f1":lambda c,m:train_stage_b(c,m,"full","f1_ce"),
        "linear_ce":lambda c,m:train_stage_b(c,m,"linear","f2_ce"),"full_ce":lambda c,m:train_stage_b(c,m,"full","f2_ce"),
        "linear_kd":lambda c,m:train_stage_b(c,m,"linear","f2_kd"),"full_kd":lambda c,m:train_stage_b(c,m,"full","f2_kd"),"evaluate":final_evaluate,
    }
    actions[args.action](cfg,args.mode)


if __name__=="__main__": main()
