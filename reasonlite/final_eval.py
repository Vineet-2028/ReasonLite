"""Frozen four-arm evaluation with atomic batch checkpoints. No training."""
import argparse, hashlib, json, math, os, platform, time, statistics
from pathlib import Path
from fractions import Fraction
from .data import load_rows, messages, digest, dump_json
from .scoring import grade, SCORING_VERSION
from .evaluate import summarize

ARMS = {'E0':None,'E1':'runs/e1_sft_v2/adapter','E2':'runs/e2_grpo_correctness_v1/adapter','E3':'runs/e3_grpo_efficiency_v1/adapter'}
VAL = dict(zip(ARMS,['runs/e0_validation_v2','runs/e1_validation_v2','runs/e2_validation_v1','runs/e3_validation_v1']))
CAP=384

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def lines(p):
    with Path(p).open(encoding='utf-8') as f: return [json.loads(l) for l in f]
def atomic(p,obj):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp'); dump_json(tmp,obj); os.replace(tmp,p)
def completion(ids, eos):
    for i,t in enumerate(ids):
        if t in eos: return ids[:i+1], i+1==CAP
    return ids, len(ids)==CAP

def verify_records(records, rows):
    assert len(records)==len(rows)
    for r,g in zip(records,rows):
        assert (r['id'],r['gold'])==(g['id'],g['gold'])
        assert all(r[k]==v for k,v in grade(r['output'],Fraction(g['gold']),truncated=r['hit_token_limit']).items())
        assert 0<r['tokens']<=CAP

def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--split',choices=['validation','test'],required=True); p.add_argument('--root',default='.'); p.add_argument('--out',required=True); a=p.parse_args()
    root=Path(a.root).resolve(); out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=read(root/'runs/final_protocol_v1.json'); assert digest(root/'reasonlite/scoring.py')==lock['scorer_sha256']; assert digest(Path(__file__))==lock['evaluator_sha256']; assert SCORING_VERSION==lock['scoring_version']
    manifest, rows=load_rows(root/'data/prepared',a.split); assert digest(root/'data/prepared/manifest.json')==lock['manifest_sha256']; assert manifest['system_prompt']==messages('')[0]['content']
    B=lock['batch_size']; assert lock['arms']==ARMS
    if a.split=='validation': rows=rows[:16]
    else:
        assert len(rows)==1319
        for arm in ARMS:
            v=read(root/'runs/final_v1/calibration'/arm/'metrics.json'); assert v['protocol_sha256']==digest(root/'runs/final_protocol_v1.json') and v['n']==16 and v['arm']==arm and v['split']=='validation'
    import torch, transformers, peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert transformers.__version__=='4.51.3' and peft.__version__=='0.15.2'
    set_seed(manifest['seed']); tok=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['model_revision'],trust_remote_code=False); tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['model_revision'],torch_dtype=torch.float16,trust_remote_code=False).to('cuda')
    adapter=ARMS[a.arm]
    if adapter:
        ap=root/adapter; assert digest(ap/'adapter_model.safetensors')==lock['adapter_sha256'][a.arm]
        meta=read(ap/'reasonlite_training.json'); assert meta['model_revision']==manifest['model_revision'] and meta['data_manifest_sha256']==lock['manifest_sha256']
        model=peft.PeftModel.from_pretrained(model,str(ap))
    model.eval(); eos=model.generation_config.eos_token_id; eos=[eos] if isinstance(eos,int) else list(eos or [])
    assert eos and tok.pad_token_id is not None
    protocol={'arm':a.arm,'split':a.split,'n':len(rows),'split_sha256':manifest['files'][a.split]['sha256'],'protocol_sha256':digest(root/'runs/final_protocol_v1.json'),'model_revision':manifest['model_revision'],'adapter_sha256':lock['adapter_sha256'][a.arm],'scoring_version':SCORING_VERSION,'batch_size':B,'max_new_tokens':CAP,'decoding':'greedy','dtype':'torch.float16','device':'cuda','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__,'python':platform.python_version(),'eos_token_ids':eos,'pad_token_id':tok.pad_token_id,'attention_implementation':model.config._attn_implementation,'timing':'synchronized batch wall time; per-example seconds are amortized, not request latency'}
    if a.split=='test':
        for arm in ARMS:
            v=read(root/'runs/final_v1/calibration'/arm/'metrics.json')
            for k in ('gpu','torch','transformers','peft','python','dtype','batch_size','attention_implementation','eos_token_ids','pad_token_id','max_new_tokens'):
                assert v[k]==protocol[k], f'Calibration environment differs: {arm}/{k}'
    pp=out/'protocol.json'
    if pp.exists(): assert read(pp)==protocol, 'Resume environment or protocol differs; use a fresh run directory for all arms'
    else: atomic(pp,protocol)
    @torch.inference_mode()
    def generate(questions,cap):
        prompts=[tok.apply_chat_template(messages(q),tokenize=False,add_generation_prompt=True) for q in questions]
        inputs=tok(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
        torch.cuda.synchronize(); start=time.perf_counter()
        result=model.generate(**inputs,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id,temperature=None,top_p=None,top_k=None)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        return result[:,inputs['input_ids'].shape[1]:].tolist(),elapsed
    generate(['What is 2 plus 2?']*B,16); torch.cuda.reset_peak_memory_stats()
    allrows=[]; batchtimes=[]; maxmem=[]
    for start in range(0,len(rows),B):
        subset=rows[start:start+B]; bp=out/f'batch_{start:05d}.json'
        if bp.exists():
            batch=read(bp); assert batch['protocol_sha256']==digest(pp); verify_records(batch['rows'],subset)
        else:
            ids,elapsed=generate([r['question'] for r in subset],CAP); records=[]
            for g,seq in zip(subset,ids):
                seq,hit=completion(seq,eos); text=tok.decode(seq,skip_special_tokens=True)
                records.append({'id':g['id'],'gold':g['gold'],'output':text,'tokens':len(seq),'seconds':elapsed/len(subset),'hit_token_limit':hit,**grade(text,Fraction(g['gold']),truncated=hit)})
            verify_records(records,subset)
            batch={'protocol_sha256':digest(pp),'rows':records,'seconds':elapsed,'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20}; atomic(bp,batch)
        allrows.extend(batch['rows']); batchtimes.append(batch['seconds']); maxmem.append(batch['peak_allocated_gpu_mib']); print(f'{a.arm} {a.split}: {len(allrows)}/{len(rows)}',flush=True)
    verify_records(allrows,rows); m=summarize(allrows); m['amortized_seconds_per_question']=m.pop('mean_latency_s'); m['p95_amortized_seconds_per_question']=m.pop('p95_latency_s'); m.update(protocol); m.update({'generation_wall_seconds':sum(batchtimes),'questions_per_second':len(rows)/sum(batchtimes),'peak_allocated_gpu_mib':max(maxmem)})
    tmp=out/'predictions.jsonl.tmp'
    with tmp.open('w',encoding='utf-8') as f:
        for r in allrows: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(tmp,out/'predictions.jsonl'); m['predictions_sha256']=digest(out/'predictions.jsonl'); atomic(out/'metrics.json',m); print(json.dumps(m,indent=2))

if __name__=='__main__': main()
