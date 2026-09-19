import json
import math
import random
import time

import torch

from common import load_model, log, manifests, read_json, run_root, save_json, save_tensor
from modules import CrossWriter, SlotCompressor, full_kl, representation_loss
from protocol import final_logits, load_native


def checkpoint(cfg, stage, which='best'):
    return run_root(cfg) / 'checkpoints' / stage / (which + '.pt')


def save_checkpoint(cfg, stage, which, module, epoch, loss):
    save_tensor(checkpoint(cfg, stage, which), {'signature': cfg['signature'], 'state': module.state_dict(),
                                             'epoch': epoch, 'validation_loss': loss})


def restore(cfg, module, stage, which='best'):
    p = torch.load(checkpoint(cfg, stage, which), map_location='cpu', weights_only=True)
    if p['signature'] != cfg['signature']: raise RuntimeError('Checkpoint config mismatch')
    module.load_state_dict(p['state']); return p


def self_module(cfg, family):
    m = SlotCompressor(36 if family == 'qwen' else 28, cfg['slots'])
    restore(cfg, m, 'self_' + family)
    return m.cuda().eval()


def slot_path(cfg, family, split, row):
    return run_root(cfg) / 'cache' / (family + '_slots') / split / (row['id'] + '.pt')


def slots(cfg, family, split, row):
    obj = torch.load(slot_path(cfg, family, split, row), map_location='cpu', weights_only=True)
    if obj['signature'] != cfg['signature']: raise RuntimeError('Slot cache config mismatch')
    return obj


def objective(cfg, stage, module, model, row, split):
    if stage.startswith('self_'):
        family = stage[5:]
        data = load_native(cfg, family, split, row)
        k, v = module(data['k'].cuda(), data['v'].cuda())
        logits = final_logits(model, row['encoded'][family]['suffix'], k, v)
        return full_kl(logits, data['native_logits'].cuda(), cfg['temperature'])
    source = load_native(cfg, 'llama', split, row)
    k, v = module(source['k'].cuda(), source['v'].cuda())
    target = slots(cfg, 'qwen', split, row)
    if stage == 'stage_a':
        return representation_loss(k, v, target['k'].cuda(), target['v'].cuda())[0]
    logits = final_logits(model, row['encoded']['qwen']['suffix'], k, v)
    return full_kl(logits, target['logits'].cuda(), cfg['temperature'])


@torch.no_grad()
def validate(cfg, stage, module, model, rows):
    module.eval()
    values = [objective(cfg, stage, module, model, row, 'validation').item() for row in rows]
    if not all(math.isfinite(v) for v in values): raise RuntimeError('Nonfinite validation')
    module.train(); return sum(values) / len(values)


def train(cfg, stage):
    train_rows, val_rows = manifests(cfg, 'train'), manifests(cfg, 'validation')
    if stage.startswith('self_'):
        family = stage[5:]
        module = SlotCompressor(36 if family == 'qwen' else 28, cfg['slots']).cuda()
        model = load_model(cfg, family)
        epochs, lr = cfg['self_epochs'], cfg['self_lr']
    else:
        module = CrossWriter(cfg['slots']).cuda()
        if stage == 'stage_a':
            payload = torch.load(checkpoint(cfg, 'self_llama'), map_location='cpu', weights_only=True)
            module.compressor.load_state_dict(payload['state'])
            model = None
        else:
            restore(cfg, module, 'stage_a'); model = load_model(cfg, 'qwen')
        epochs, lr = cfg[stage + '_epochs'], cfg[stage + '_lr']
    params = list(module.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    destination = run_root(cfg) / 'training' / stage; destination.mkdir(parents=True, exist_ok=True)
    initial = validate(cfg, stage, module, model, val_rows)
    save_checkpoint(cfg, stage, 'initial', module, 0, initial)
    save_checkpoint(cfg, stage, 'best', module, 0, initial)
    best, best_epoch, best_trained = initial, 0, float('inf')
    history, epoch_rows, exposures, steps, clip_count = [], [], 0, 0, 0
    started = time.monotonic()
    with (destination / 'steps.jsonl').open('w', encoding='utf-8') as out:
        for epoch in range(1, epochs + 1):
            order = list(range(len(train_rows))); random.Random(cfg['seed'] + epoch).shuffle(order)
            if len(set(order)) != len(train_rows): raise RuntimeError('Invalid epoch sampler')
            epoch_losses = []
            for begin in range(0, len(order), cfg['batch_size']):
                batch = order[begin:begin + cfg['batch_size']]; optimizer.zero_grad(set_to_none=True)
                losses = []
                for idx in batch:
                    loss = objective(cfg, stage, module, model, train_rows[idx], 'train')
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                    (loss / len(batch)).backward(); losses.append(loss.item())
                missing = [name for name, p in module.named_parameters() if p.grad is None]
                if missing: raise RuntimeError(f'Missing gradients: {missing}')
                norm = torch.nn.utils.clip_grad_norm_(params, cfg['clip'])
                if not torch.isfinite(norm): raise RuntimeError('Nonfinite Writer gradient')
                if model is not None and any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Frozen receiver has gradients')
                optimizer.step(); steps += 1; exposures += len(batch)
                clipped = norm.item() > cfg['clip']; clip_count += int(clipped)
                entry = {'epoch': epoch, 'step': steps, 'loss': sum(losses) / len(losses),
                         'pre_clip_grad_norm': norm.item(), 'clipped': clipped,
                         'sample_ids': [train_rows[i]['id'] for i in batch]}
                out.write(json.dumps(entry) + '\n'); out.flush(); history.append(entry); epoch_losses.extend(losses)
                log(f'{stage} epoch={epoch}/{epochs} step={steps} loss={entry["loss"]:.6f}')
            score = validate(cfg, stage, module, model, val_rows)
            save_checkpoint(cfg, stage, f'epoch_{epoch}', module, epoch, score)
            save_checkpoint(cfg, stage, 'last', module, epoch, score)
            if score < best_trained:
                best_trained = score; save_checkpoint(cfg, stage, 'best_trained', module, epoch, score)
            if score < best:
                best, best_epoch = score, epoch; save_checkpoint(cfg, stage, 'best', module, epoch, score)
            epoch_rows.append({'epoch': epoch, 'validation_loss': score, 'mean_train_loss': sum(epoch_losses) / len(epoch_losses)})
            save_json(destination / 'epochs.json', epoch_rows)
            log(f'{stage} validation epoch={epoch} loss={score:.6f}; best_epoch={best_epoch}')
    save_json(destination / 'summary.json', {'stage': stage, 'initial_validation_loss': initial,
              'best_validation_loss': best, 'best_epoch': best_epoch, 'best_trained_validation_loss': best_trained,
              'epochs': epochs, 'optimizer_steps': steps, 'sample_exposures': exposures, 'clip_rate': clip_count / steps,
              'trainable_parameters': sum(p.numel() for p in params), 'training_uses_gold': False,
              'true_epoch_sampler': True, 'receiver_frozen': True, 'seconds': time.monotonic() - started})
    del module, model, optimizer
    torch.cuda.empty_cache()


@torch.no_grad()
def materialize_slots(cfg, family):
    module, model = self_module(cfg, family), load_model(cfg, family)
    selected = torch.load(checkpoint(cfg, 'self_' + family), map_location='cpu', weights_only=True)
    for split in cfg['splits']:
        rows = manifests(cfg, split)
        for i, row in enumerate(rows):
            data = load_native(cfg, family, split, row)
            k, v = module(data['k'].cuda(), data['v'].cuda())
            logits = final_logits(model, row['encoded'][family]['suffix'], k, v)
            save_tensor(slot_path(cfg, family, split, row), {'signature': cfg['signature'],
                        'k': k.cpu(), 'v': v.cpu(), 'logits': logits.cpu(), 'compressor_epoch': selected['epoch']})
            log(f'{family} slots {split} {i + 1}/{len(rows)}')
    del module, model
    torch.cuda.empty_cache()


def prediction(logits, row, family):
    return int(logits[row['encoded'][family]['choice_ids']].argmax())


def result_row(row, condition, family, logits, native, reference=None, refname=None):
    ref = native if reference is None else reference
    p = prediction(logits, row, family)
    return {'id': row['id'], 'category': row['category'], 'condition': condition, 'receiver': family,
            'gold_index': row['gold_index'], 'prediction_index': p, 'prediction': 'ABCDEFGHIJ'[p],
            'accuracy': float(p == row['gold_index']), 'native_agreement': float(p == prediction(native, row, family)),
            'reference_agreement': float(p == prediction(ref, row, family)),
            'reference': refname or family + '_native', 'full_kl_vs_native': full_kl(logits, native).item(),
            'full_kl_vs_reference': full_kl(logits, ref).item(),
            'choice_logits': logits[row['encoded'][family]['choice_ids']].cpu().tolist()}


@torch.no_grad()
def evaluate(cfg):
    rows = manifests(cfg, 'test', labels=True); records = []
    for row in rows:
        for family in ('qwen', 'llama'):
            native = load_native(cfg, family, 'test', row)['native_logits']
            slot = slots(cfg, family, 'test', row)['logits']
            records.append(result_row(row, family + '_native', family, native, native))
            records.append(result_row(row, family + '_self_slot', family, slot, native))
    model = load_model(cfg, 'qwen')
    for row in rows:
        native = load_native(cfg, 'qwen', 'test', row)['native_logits'].cuda()
        logits = final_logits(model, row['encoded']['qwen']['question_only'])
        records.append(result_row(row, 'qwen_question_only', 'qwen', logits, native))
    for stage, which in [('stage_a', 'best'), ('stage_b', 'best'), ('stage_b', 'best_trained'), ('stage_b', 'last')]:
        module = CrossWriter(cfg['slots']).cuda().eval(); cp = restore(cfg, module, stage, which)
        for i, row in enumerate(rows):
            src = load_native(cfg, 'llama', 'test', row)
            native = load_native(cfg, 'qwen', 'test', row)['native_logits'].cuda()
            anchor = slots(cfg, 'qwen', 'test', row)
            k, v = module(src['k'].cuda(), src['v'].cuda())
            logits = final_logits(model, row['encoded']['qwen']['suffix'], k, v)
            record = result_row(row, stage + '_' + which, 'qwen', logits, native, anchor['logits'].cuda(), 'qwen_self_slot')
            record['checkpoint_epoch'] = cp['epoch']
            record['representation_loss'], record['per_layer'] = representation_loss(k, v, anchor['k'].cuda(), anchor['v'].cuda())
            record['representation_loss'] = record['representation_loss'].item()
            records.append(record)
            if stage == 'stage_b' and which == 'best':
                zero = final_logits(model, row['encoded']['qwen']['suffix'], torch.zeros_like(k), torch.zeros_like(v))
                records.append(result_row(row, 'zero_slots', 'qwen', zero, native, anchor['logits'].cuda(), 'qwen_self_slot'))
                donor = rows[(i + 1) % len(rows)]
                if donor['id'] == row['id']: raise RuntimeError('Shuffled donor must differ')
                ds = load_native(cfg, 'llama', 'test', donor)
                dk, dv = module(ds['k'].cuda(), ds['v'].cuda())
                shuffled = final_logits(model, row['encoded']['qwen']['suffix'], dk, dv)
                r = result_row(row, 'shuffled_llama_slots', 'qwen', shuffled, native, anchor['logits'].cuda(), 'qwen_self_slot')
                r['donor_id'] = donor['id']; records.append(r)
            log(f'evaluate {stage}/{which} {i + 1}/{len(rows)}')
        del module
    out = run_root(cfg) / 'results'; out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_sample_predictions.jsonl').open('w', encoding='utf-8') as f:
        for row in records: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    summary = {}
    for condition in sorted({r['condition'] for r in records}):
        subset = [r for r in records if r['condition'] == condition]
        summary[condition] = {'count': len(subset), 'reference': subset[0]['reference'], **{
            key: sum(x[key] for x in subset) / len(subset) for key in
            ('accuracy', 'native_agreement', 'reference_agreement', 'full_kl_vs_native', 'full_kl_vs_reference')}}
    acc = lambda c: summary[c]['accuracy']
    ratio = lambda a, b: acc(a) / acc(b) if acc(b) > 0 else None
    save_json(out / 'summary.json', {'conditions': summary, 'retention': {
        'qwen': ratio('qwen_self_slot', 'qwen_native'), 'llama': ratio('llama_self_slot', 'llama_native'),
        'cross_stage_a': ratio('stage_a_best', 'qwen_self_slot'), 'cross_stage_b': ratio('stage_b_best', 'qwen_self_slot')},
        'accuracy_gaps': {'qwen_compression': acc('qwen_native') - acc('qwen_self_slot'),
                          'llama_compression': acc('llama_native') - acc('llama_self_slot'),
                          'cross_registration': acc('qwen_self_slot') - acc('stage_b_best')},
        'zero_denominator_retention': None, 'benchmark': 'internal held-out feasibility subset',
        'position_policy': 'pre-RoPE pooling; receiver RoPE slots=0..63, suffix starts at 64',
        'short_prefix_policy': 'keep all examples, including T<64; in that case this is resampling not compression'})
    log('Evaluation files saved')
