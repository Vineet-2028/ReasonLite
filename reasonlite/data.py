"""Prepare reproducible splits, with official test excluded from development."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from .scoring import reference

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
DATASET = 'openai/gsm8k'
SYSTEM = 'Solve the math problem. Show your reasoning, then end with #### <number> on its own line.'

def messages(question):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': question}]

def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def choose(n, seed, sft, rl, val):
    if min(sft, rl, val) <= 0 or sft+rl+val > n:
        raise ValueError('Invalid split sizes')
    ids = list(range(n)); random.Random(seed).shuffle(ids)
    return {'validation': ids[:val], 'sft': ids[val:val+sft], 'rl': ids[val+sft:val+sft+rl]}

def write_rows(path, source, ids, origin):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for i in ids:
            row = source[i]
            gold = reference(row['answer'])
            f.write(json.dumps({'id': f'{origin}:{i}', 'question': row['question'],
                                'solution': row['answer'], 'gold': str(gold)}, ensure_ascii=False)+'\n')

def load_rows(root, split):
    root = Path(root); manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    file = root/f'{split}.jsonl'
    if digest(file) != manifest['files'][split]['sha256']:
        raise ValueError(f'Split {split} differs from manifest')
    with file.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != manifest['files'][split]['count'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Split integrity failure')
    return manifest, rows

def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', default='data/prepared'); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sft', type=int, default=512); p.add_argument('--rl', type=int, default=256); p.add_argument('--val', type=int, default=128)
    p.add_argument('--model-revision'); p.add_argument('--dataset-revision'); p.add_argument('--include-test', action='store_true')
    a = p.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    api = HfApi(); mrev = api.model_info(MODEL, revision=a.model_revision).sha; drev = api.dataset_info(DATASET, revision=a.dataset_revision).sha
    if not mrev or not drev: raise RuntimeError('Cannot resolve immutable revisions')
    train = load_dataset(DATASET, 'main', split='train', revision=drev)
    groups = choose(len(train), a.seed, a.sft, a.rl, a.val)
    manifest = {'seed': a.seed, 'model': MODEL, 'model_revision': mrev, 'dataset': DATASET, 'dataset_revision': drev,
                'system_prompt': SYSTEM, 'files': {}, 'official_test_included': a.include_test}
    for split, ids in groups.items():
        path = out/f'{split}.jsonl'; write_rows(path, train, ids, 'train')
        manifest['files'][split] = {'count': len(ids), 'sha256': digest(path)}
    if a.include_test:
        test = load_dataset(DATASET, 'main', split='test', revision=drev)
        path = out/'test.jsonl'; write_rows(path, test, range(len(test)), 'test')
        manifest['files']['test'] = {'count': len(test), 'sha256': digest(path)}
    dump_json(out/'manifest.json', manifest); print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
