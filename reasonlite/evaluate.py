"""Greedy, single-example evaluation with per-example outputs and provenance."""
import argparse
import json
import math
import platform
import statistics
import time
from fractions import Fraction
from pathlib import Path
from .data import load_rows, messages, digest, dump_json
from .scoring import grade

def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)]

def summarize(rows):
    n = len(rows)
    if not n: raise ValueError('Empty evaluation')
    correct = sum(r['correct'] for r in rows)
    tokens = sum(r['tokens'] for r in rows)
    accuracy = correct/n; z = 1.95996398454
    denom = 1+z*z/n; center = (accuracy+z*z/(2*n))/denom
    radius = z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n))/denom
    return {'n': n, 'correct': correct, 'accuracy': accuracy, 'wilson_95': [max(0,center-radius),min(1,center+radius)],
            'strict_accuracy': sum(r['strict_correct'] for r in rows)/n,
            'invalid_rate': 1-sum(r['valid'] for r in rows)/n, 'format_rate': sum(r['format_ok'] for r in rows)/n,
            'mean_tokens': tokens/n, 'correct_per_1000_generated_tokens': 1000*correct/max(1,tokens),
            'mean_latency_s': statistics.mean(r['seconds'] for r in rows), 'p95_latency_s': percentile([r['seconds'] for r in rows], .95),
            'token_limit_rate': sum(r['hit_token_limit'] for r in rows)/n}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', default='data/prepared'); p.add_argument('--split', choices=['validation','test'], default='validation')
    p.add_argument('--limit', type=int, default=128); p.add_argument('--max-new-tokens', type=int, default=384)
    p.add_argument('--adapter'); p.add_argument('--out', required=True); p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto'); p.add_argument('--final-test', action='store_true')
    a = p.parse_args()
    if a.limit <= 0 or a.max_new_tokens <= 0: p.error('Limits must be positive')
    if a.split == 'test' and not a.final_test: p.error('Official test requires --final-test after protocol freeze')
    manifest, rows = load_rows(a.data, a.split); rows = rows[:a.limit]
    if a.split == 'test' and len(rows) != manifest['files']['test']['count']: p.error('Final evaluation must include complete official test')
    if a.adapter:
        meta = json.loads((Path(a.adapter)/'reasonlite_training.json').read_text(encoding='utf-8'))
        if meta['model_revision'] != manifest['model_revision'] or meta['data_manifest_sha256'] != digest(Path(a.data)/'manifest.json'):
            raise ValueError('Adapter and evaluation provenance differ')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    set_seed(manifest['seed']); device = 'cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device == 'auto' else a.device)
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    tok = AutoTokenizer.from_pretrained(manifest['model'], revision=manifest['model_revision'], trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], revision=manifest['model_revision'], torch_dtype=dtype, trust_remote_code=False).to(device)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval(); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    def sync():
        if device == 'cuda': torch.cuda.synchronize()
    @torch.inference_mode()
    def generate(question, cap):
        prompt = tok.apply_chat_template(messages(question), tokenize=False, add_generation_prompt=True)
        batch = tok(prompt, return_tensors='pt', add_special_tokens=False).to(device)
        sync(); start = time.perf_counter()
        result = model.generate(**batch, max_new_tokens=cap, do_sample=False, pad_token_id=tok.pad_token_id,
                                temperature=None, top_p=None, top_k=None)
        sync(); elapsed = time.perf_counter()-start
        ids = result[0, batch['input_ids'].shape[1]:]; text = tok.decode(ids, skip_special_tokens=True)
        return text, len(ids), elapsed
    generate('What is 2 plus 2?', 16)
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    results = []
    with (out/'predictions.jsonl').open('w', encoding='utf-8') as f:
        for i,row in enumerate(rows):
            text, count, elapsed = generate(row['question'], a.max_new_tokens)
            r = {'id':row['id'], 'gold':row['gold'], 'output':text, 'tokens':count, 'seconds':elapsed,
                 'hit_token_limit':count == a.max_new_tokens, **grade(text,Fraction(row['gold']))}
            results.append(r); f.write(json.dumps(r, ensure_ascii=False)+'\n'); f.flush()
            if (i+1)%10 == 0 or i+1 == len(rows): print(f'{i+1}/{len(rows)}', flush=True)
    metrics = summarize(results)
    metrics.update({'model':manifest['model'], 'model_revision':manifest['model_revision'], 'adapter':a.adapter,
                    'adapter_sha256':digest(Path(a.adapter)/'adapter_model.safetensors') if a.adapter else None,
                    'split':a.split, 'split_sha256':manifest['files'][a.split]['sha256'], 'data_manifest_sha256':digest(Path(a.data)/'manifest.json'),
                    'max_new_tokens':a.max_new_tokens, 'decoding':'greedy', 'batch_size':1, 'warmup':1,
                    'device':device, 'gpu':torch.cuda.get_device_name() if device=='cuda' else None, 'dtype':str(dtype),
                    'python':platform.python_version(), 'torch':torch.__version__, 'transformers':transformers.__version__,
                    'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None})
    dump_json(out/'metrics.json', metrics); print(json.dumps(metrics, indent=2))

if __name__ == '__main__': main()
