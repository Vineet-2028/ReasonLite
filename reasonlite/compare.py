"""Compare only runs with matching evaluation protocols and ordered IDs."""
import argparse
import json
from pathlib import Path
from .data import dump_json

KEYS = ['scoring_version','model_revision','split','split_sha256','data_manifest_sha256','n','max_new_tokens','decoding','batch_size','warmup','device','gpu','dtype','torch','transformers']
def compare(before, after):
    def read(root):
        root = Path(root)
        with (root/'predictions.jsonl').open(encoding='utf-8') as f:
            predictions = [json.loads(line) for line in f]
        return json.loads((root/'metrics.json').read_text()), predictions
    b, br = read(before); a, ar = read(after)
    for k in KEYS:
        if b[k] != a[k]: raise ValueError(f'Evaluation mismatch: {k}')
    if [(r['id'],r['gold']) for r in br] != [(r['id'],r['gold']) for r in ar] or len(br) != b['n']:
        raise ValueError('Evaluation IDs or gold values differ')
    return {'before':str(before),'after':str(after),'n':b['n'],
            'accuracy_delta_percentage_points':100*(a['accuracy']-b['accuracy']),
            'strict_accuracy_delta_percentage_points':100*(a['strict_accuracy']-b['strict_accuracy']),
            'mean_tokens_delta':a['mean_tokens']-b['mean_tokens'],
            'wrong_to_correct':sum(not x['correct'] and y['correct'] for x,y in zip(br,ar)),
            'correct_to_wrong':sum(x['correct'] and not y['correct'] for x,y in zip(br,ar)),
            'both_correct':sum(x['correct'] and y['correct'] for x,y in zip(br,ar)),
            'both_wrong':sum(not x['correct'] and not y['correct'] for x,y in zip(br,ar))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('before'); p.add_argument('after'); p.add_argument('--out',required=True); a=p.parse_args()
    result=compare(a.before,a.after); dump_json(a.out,result); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
