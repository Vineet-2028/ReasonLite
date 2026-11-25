"""CPU-only verification of frozen evidence. Run from repository root."""
import argparse, hashlib, json, math, struct
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
from itertools import combinations
from reasonlite.data import load_rows, digest
from reasonlite.final_eval import ARMS, VAL, verify_records, lines, read
from reasonlite.evaluate import summarize
from reasonlite.final_report import paired, holm, audit
from reasonlite.train_grpo import shape

def check_close(x,y):
    if isinstance(x, list):
        assert len(x)==len(y)
        for a,b in zip(x,y): check_close(a,b)
    elif isinstance(x,(int,float)) and not isinstance(x,bool): assert math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-12),(x,y)
    else: assert x==y,(x,y)

def safe_tensor_finite(p):
    raw=p.read_bytes(); n=struct.unpack_from('<Q',raw)[0]; header=json.loads(raw[8:8+n]); body=memoryview(raw)[8+n:]; total=0
    for name,v in header.items():
        if name=='__metadata__': continue
        dtype=v['dtype']; a,b=v['data_offsets']; assert 0<=a<=b<=len(body)
        count=math.prod(v['shape']); fmt={'F32':'f','F16':'e','F64':'d'}[dtype]; assert b-a==count*struct.calcsize(fmt)
        assert all(math.isfinite(x[0]) for x in struct.iter_unpack('<'+fmt,body[a:b])),(p,name)
        total+=count
    return total

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--strict-release',action='store_true',help='Verify every original package file, including editable documentation'); args=parser.parse_args()
    root=Path.cwd(); h=read(root/'provenance/release_hashes.json')
    selected={name:wanted for name,wanted in h.items() if args.strict_release or name.startswith(('data/','runs/','reasonlite/','configs/','provenance/'))}
    for name,wanted in selected.items(): assert digest(root/name)==wanted, name
    print('Release file hashes:',len(selected),'verified (strict release)' if args.strict_release else 'verified (frozen source/evidence)')
    lock=read(root/'runs/final_protocol_v1.json'); assert digest(root/'reasonlite/scoring.py')==lock['scorer_sha256']; assert digest(root/'reasonlite/final_eval.py')==lock['evaluator_sha256']; assert digest(root/'data/prepared/manifest.json')==lock['manifest_sha256']
    splits={k:load_rows(root/'data/prepared',k)[1] for k in ('sft','rl','validation','test')}
    assert {k:len(v) for k,v in splits.items()}=={'sft':512,'rl':256,'validation':128,'test':1319}
    for a,b in combinations(splits,2): assert not ({r['id'] for r in splits[a]}&{r['id'] for r in splits[b]})
    print('Frozen splits verified and disjoint')
    report=read(root/'runs/final_v1/report/summary.json'); predictions={}
    for arm,adapter in ARMS.items():
        if adapter:
            p=root/adapter/'adapter_model.safetensors'; assert digest(p)==lock['adapter_sha256'][arm]; assert safe_tensor_finite(p)==1081344
            meta=read(root/adapter/'reasonlite_training.json'); assert meta['data_manifest_sha256']==lock['manifest_sha256']; assert meta['model_revision']==report['metrics'][arm]['model_revision'] if 'model_revision' in report['metrics'][arm] else meta['model_revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
        p=root/'runs/final_v1/test'/arm; m=read(p/'metrics.json'); assert m==report['metrics'][arm]; assert digest(p/'predictions.jsonl')==m['predictions_sha256']; assert m['adapter_sha256']==lock['adapter_sha256'][arm]; assert m['protocol_sha256']==digest(root/'runs/final_protocol_v1.json'); assert m['n']==1319 and m['arm']==arm and m['split']=='test'
        rows=lines(p/'predictions.jsonl'); verify_records(rows,splits['test']); predictions[arm]=rows
        stats=summarize(rows)
        for k,v in stats.items(): check_close(v,m[{'mean_latency_s':'amortized_seconds_per_question','p95_latency_s':'p95_amortized_seconds_per_question'}.get(k,k)])
        check_close(m['questions_per_second'],len(rows)/sum(r['seconds'] for r in rows))
        v=lines(root/VAL[arm]/'predictions.jsonl'); verify_records(v,splits['validation']); vm=read(root/VAL[arm]/'metrics.json')
        for k,val in summarize(v).items(): check_close(val,vm[k])
        c=root/'runs/final_v1/calibration'/arm; cr=lines(c/'predictions.jsonl'); cm=read(c/'metrics.json'); assert digest(c/'predictions.jsonl')==cm['predictions_sha256']; verify_records(cr,splits['validation'][:16]); assert cm['n']==16
        print(arm, 'test',m['correct'],'/',m['n'],'accuracy',round(m['accuracy'],6),'tokens',round(m['mean_tokens'],3))
    comp=[{'before':a,'after':b,**paired(predictions[a],predictions[b])} for a,b in combinations(ARMS,2)]
    for c,p in zip(comp,holm([c['mcnemar_p'] for c in comp])): c['holm_p']=p
    for a,b in zip(comp,report['comparisons']):
        for k in a: check_close(a[k],b[k])
    assert len(comp)==len(report['comparisons'])==6
    assert audit(root)==report['grpo_diagnostics']
    a=root/'runs/e2_grpo_correctness_v1'; b=root/'runs/e3_grpo_efficiency_v1'; m2=read(a/'train_metrics.json'); m3=read(b/'train_metrics.json')
    for k in ('parent_adapter_sha256','prompt_ids','configuration','seed','optimizer_steps','rl_split_sha256','scoring_version'): assert m2[k]==m3[k], k
    assert m2['parent_adapter_sha256']==lock['adapter_sha256']['E1']; assert m2['prompt_ids']==[r['id'] for r in splits['rl'][:64]]
    for folder,meta,mode in ((a,m2,'correctness'),(b,m3,'efficiency')):
        rr=lines(folder/'rollouts.jsonl'); groups=defaultdict(list); gold={r['id']:r['gold'] for r in splits['rl'][:64]}
        assert len(rr)==256
        for r in rr:
            assert gold[r['id']]==r['gold']; val,g=shape(r['output'],r['gold'],r['retokenized_tokens'],mode); check_close(val,r['reward']); assert all(r[k]==v for k,v in g.items()); groups[r['id']].append(r)
        assert set(groups)==set(gold) and all(len(g)==4 for g in groups.values()); assert sum(r['correct'] for r in rr)==meta['rollout_counts']['correct']
    print('PASS: 5,276 held-out predictions, 512 validation predictions, 64 calibration predictions, 512 RL rollouts, 6 paired comparisons and 3 final adapters verified.')
if __name__=='__main__': main()
