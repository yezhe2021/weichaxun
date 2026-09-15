from collections import defaultdict

import pyarrow.parquet as pq

from common import digest, log, run_root, save_json, tokenizer

LABELS = 'ABCDEFGHIJ'


def texts(row):
    prefix = 'Candidate answers:\n' + '\n'.join(f'{LABELS[i]}. {x}' for i, x in enumerate(row['options'])) + '\n\n'
    suffix = f"Question:\n{row['question']}\n\nChoose the best answer. Reply with only one letter from A to {LABELS[len(row['options']) - 1]}.\nAnswer:"
    return prefix, suffix


def encode(tok, prefix, suffix, option_count):
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    p, s = enc(prefix), enc(suffix)
    if p + s != enc(prefix + suffix):
        raise RuntimeError('Noncompositional options/question tokenizer boundary')
    choice_ids = []
    for letter in LABELS[:option_count]:
        # Require exactly the continuation token that extends the serialized Answer: suffix.
        full = enc(suffix + ' ' + letter)
        if full[:len(s)] != s or len(full) != len(s) + 1:
            raise RuntimeError(f'Choice {letter} is not a single stable continuation token')
        choice_ids.append(full[-1])
    if len(set(choice_ids)) != len(choice_ids): raise RuntimeError('Duplicate choice token ids')
    # Family-native BOS appears once at the beginning, never in the cached suffix.
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return {'prefix': bos + p, 'suffix': s, 'full': bos + p + s, 'question_only': bos + s, 'choice_ids': choice_ids}


def prepare(cfg):
    toks = {f: tokenizer(path) for f, path in cfg['models'].items()}
    rows = pq.read_table(cfg['data_file'], columns=['question_id', 'question', 'options', 'answer_index', 'category']).to_pylist()
    groups = defaultdict(list)
    for row in rows: groups[row['category']].append(row)
    for cat in groups:
        groups[cat].sort(key=lambda r: digest([cfg['seed'], r['question_id']]))
    selected, seen_ids, seen_content, rejected = [], set(), set(), 0
    required = sum(cfg['splits'].values())
    while len(selected) < required and any(groups.values()):
        for cat in sorted(groups):
            if not groups[cat] or len(selected) >= required: continue
            row = groups[cat].pop(); content = digest([row['question'], row['options']])
            if str(row['question_id']) in seen_ids or content in seen_content: continue
            if not 2 <= len(row['options']) <= 10: raise RuntimeError('Invalid option count')
            prefix, suffix = texts(row)
            encoded = {f: encode(tok, prefix, suffix, len(row['options'])) for f, tok in toks.items()}
            if any(len(x['prefix']) > cfg['max_prefix_tokens'] or len(x['suffix']) > cfg['max_suffix_tokens'] for x in encoded.values()):
                rejected += 1; continue
            gold = int(row['answer_index'])
            if not 0 <= gold < len(row['options']): raise RuntimeError('Invalid label')
            selected.append({'id': str(row['question_id']), 'category': cat, 'question': row['question'],
                             'options': row['options'], 'gold_index': gold, 'gold_label': LABELS[gold],
                             'prefix_text': prefix, 'suffix_text': suffix, 'encoded': encoded})
            seen_ids.add(str(row['question_id'])); seen_content.add(content)
    if len(selected) != required: raise RuntimeError(f'Only {len(selected)} eligible examples')
    begin = 0
    for split, count in cfg['splits'].items():
        subset = selected[begin:begin + count]; begin += count
        save_json(run_root(cfg) / 'manifests' / f'{split}.json', {'signature': cfg['signature'], 'rows': subset})
    save_json(run_root(cfg) / 'manifests/summary.json', {
        'signature': cfg['signature'], 'counts': cfg['splits'], 'rejected_length': rejected,
        'split_source': cfg['data_file'], 'split_policy': 'category-balanced disjoint internal train/val/test from official test pool; not official benchmark score',
        'tokenizers_independent': True, 'original_options_order': True, 'sender_sees': 'options only',
        'receiver_sees': 'question plus answer instruction', 'training_uses_gold': False,
        'no_token_truncation': True, 'no_cross_family_vocab_kl': True,
        'family_bos_added_once': True})
    log(f"Prepared {cfg['splits']}; tokenizer and label continuation checks passed")
