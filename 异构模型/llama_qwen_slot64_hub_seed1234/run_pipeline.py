import argparse
import fcntl
import os
import subprocess
import sys
import traceback

from common import ROOT, configuration, digest, log, read_json, run_root, save_json, seed_all

STAGES = ('tests', 'prepare', 'cache_qwen', 'cache_llama', 'self_qwen', 'self_llama',
          'slots_qwen', 'slots_llama', 'stage_a', 'stage_b', 'evaluate')


def execute(cfg, stage):
    if stage == 'tests':
        from tests import run_tests
        save_json(run_root(cfg) / 'audit/unit_tests.json', run_tests())
    elif stage == 'prepare':
        from prepare import prepare
        prepare(cfg)
    elif stage.startswith('cache_'):
        from protocol import build
        build(cfg, stage[6:])
    elif stage.startswith('slots_'):
        from experiment import materialize_slots
        materialize_slots(cfg, stage[6:])
    elif stage == 'evaluate':
        from experiment import evaluate
        evaluate(cfg)
    else:
        from experiment import train
        train(cfg, stage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('smoke', 'pilot'), default='smoke')
    p.add_argument('--worker', choices=STAGES, help=argparse.SUPPRESS)
    args = p.parse_args(); cfg = configuration(args.mode); seed_all(cfg['seed'])
    root = run_root(cfg); root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        execute(cfg, args.worker); return
    lock = (root / 'pipeline.lock').open('w')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError(f'{args.mode} pipeline is already running')
    code_hash = digest({x.name: x.read_text(encoding='utf-8') for x in sorted(ROOT.glob('*.py'))})
    save_json(root / 'run_config.json', cfg)
    save_json(root / 'code_manifest.json', {'code_hash': code_hash, 'files': {x.name: digest(x.read_text(encoding='utf-8')) for x in ROOT.glob('*.py')}})
    (root / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    current = None
    try:
        for stage in STAGES:
            current = stage; marker = root / 'completed' / (stage + '.json')
            if marker.exists():
                saved = read_json(marker)
                if saved['signature'] != cfg['signature'] or saved['code_hash'] != code_hash:
                    raise RuntimeError('Completed stages have different config/code; use a new experiment directory')
                log(f'RESUME skip {stage}'); continue
            save_json(root / 'status.json', {'status': 'running', 'stage': stage, 'pid': os.getpid()})
            log(f'START {stage}')
            subprocess.run([sys.executable, '-u', str(ROOT / 'run_pipeline.py'), '--mode', args.mode, '--worker', stage], cwd=ROOT, check=True)
            save_json(marker, {'signature': cfg['signature'], 'code_hash': code_hash, 'stage': stage})
            log(f'DONE {stage}')
        save_json(root / 'status.json', {'status': 'completed', 'stage': 'evaluate', 'pid': os.getpid()})
        log('ALL EXPERIMENTS COMPLETED')
    except BaseException as exc:
        save_json(root / 'status.json', {'status': 'failed', 'stage': current, 'error': str(exc), 'pid': os.getpid()})
        traceback.print_exc(); raise


if __name__ == '__main__': main()
