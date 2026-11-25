"""Paired evaluation and audit reporting without model selection on test."""
import math,json,csv,itertools,statistics
from pathlib import Path
from collections import defaultdict
from .final_eval import ARMS,VAL,read,lines,verify_records
from .data import load_rows,digest,dump_json

def exact_mcnemar(b,c):
    n=b+c
    return min(1.0,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.0

def holm(p):
    order=sorted(range(len(p)),key=lambda i:p[i]); out=[0.]*len(p); running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,p[i]*(len(p)-rank))); out[i]=running
    return out

def paired(x,y):
    assert [(r['id'],r['gold']) for r in x]==[(r['id'],r['gold']) for r in y]
    b=sum(not u['correct'] and v['correct'] for u,v in zip(x,y)); c=sum(u['correct'] and not v['correct'] for u,v in zip(x,y)); n=len(x)
    return {'n':n,'wrong_to_correct':b,'correct_to_wrong':c,'accuracy_delta_pp':100*(b-c)/n,'mcnemar_p':exact_mcnemar(b,c),'mean_token_delta':statistics.mean(v['tokens']-u['tokens'] for u,v in zip(x,y))}

def audit(root):
    result={}
    for arm,path in [('E2','e2_grpo_correctness_v1'),('E3','e3_grpo_efficiency_v1')]:
        r=root/'runs'/path; logs=read(r/'training_log.json'); rolls=lines(r/'rollouts.jsonl'); groups=defaultdict(list)
        for row in rolls: groups[row['id']].append(row)
        assert len(groups)==64 and all(len(g)==4 for g in groups.values())
        result[arm]={'rollouts':len(rolls),'all_wrong_groups':sum(not any(x['correct'] for x in g) for g in groups.values()),'all_correct_groups':sum(all(x['correct'] for x in g) for g in groups.values()),'variable_reward_groups':sum(len(set(x['reward'] for x in g))>1 for g in groups.values()),'nonfinite_gradient_steps':[x.get('step') for x in logs if 'grad_norm' in x and (x['grad_norm'] is None or not math.isfinite(x['grad_norm']))],'training_seconds':read(r/'train_metrics.json')['train_seconds']}
    return result

def main():
    root=Path.cwd(); base=root/'runs/final_v1'; report=base/'report'; report.mkdir(exist_ok=True); lock=read(root/'runs/final_protocol_v1.json'); _,gold=load_rows(root/'data/prepared','test'); metrics={}; predictions={}; calibration={}
    for arm in ARMS:
        path=base/'test'/arm; m=read(path/'metrics.json'); assert m['protocol_sha256']==digest(root/'runs/final_protocol_v1.json') and m['n']==1319 and m['split']=='test' and m['arm']==arm and m['adapter_sha256']==lock['adapter_sha256'][arm]
        assert digest(path/'predictions.jsonl')==m['predictions_sha256']; r=lines(path/'predictions.jsonl'); verify_records(r,gold); metrics[arm]=m; predictions[arm]=r
        c=lines(base/'calibration'/arm/'predictions.jsonl'); old=lines(root/VAL[arm]/'predictions.jsonl')[:16]; calibration[arm]={'n':16,'exact_text_agreement':sum(x['output']==y['output'] for x,y in zip(c,old))/16,'correctness_agreement':sum(x['correct']==y['correct'] for x,y in zip(c,old))/16,'batched_accuracy':sum(x['correct'] for x in c)/16,'original_accuracy':sum(x['correct'] for x in old)/16}
    keys=['protocol_sha256','gpu','torch','transformers','peft','python','dtype','batch_size','attention_implementation','eos_token_ids','pad_token_id','max_new_tokens','split_sha256']
    for k in keys: assert all(metrics[x][k]==metrics['E0'][k] for x in ARMS),k
    comparisons=[]
    for x,y in itertools.combinations(ARMS,2): comparisons.append({'before':x,'after':y,**paired(predictions[x],predictions[y])})
    for c,p in zip(comparisons,holm([c['mcnemar_p'] for c in comparisons])): c['holm_p']=p
    diagnostics=audit(root); dump_json(report/'summary.json',{'metrics':metrics,'comparisons':comparisons,'calibration':calibration,'grpo_diagnostics':diagnostics,'primary_candidate_selected_on_validation':'E1','halo_native_status':'separate experiment; no execution claimed'})
    fields=['arm','n','correct','accuracy','strict_accuracy','format_rate','invalid_rate','mean_tokens','correct_per_1000_generated_tokens','token_limit_rate','questions_per_second','amortized_seconds_per_question','peak_allocated_gpu_mib']
    with (report/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(metrics.values())
    with (report/'paired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(comparisons[0])); w.writeheader(); w.writerows(comparisons)
    text=['# ReasonLite: frozen held-out evaluation','', 'All four pre-existing checkpoints evaluated on the same 1,319 official GSM8K test questions. E1 was selected as the efficiency candidate using development validation only. No further model selection or tuning was based on test results.','', '| Arm | Accuracy (Wilson 95% CI) | Mean tokens | Correct / 1k tokens | Format rate |','|---|---:|---:|---:|---:|']
    for arm,m in metrics.items(): text.append(f"| {arm} | {m['accuracy']:.2%} ({m['wilson_95'][0]:.2%}–{m['wilson_95'][1]:.2%}) | {m['mean_tokens']:.2f} | {m['correct_per_1000_generated_tokens']:.3f} | {m['format_rate']:.2%} |")
    text+=['','## Paired tests','', '| Contrast | Delta (pp) | Exact McNemar p | Holm-adjusted p |','|---|---:|---:|---:|']
    for c in comparisons: text.append(f"| {c['before']} → {c['after']} | {c['accuracy_delta_pp']:+.2f} | {c['mcnemar_p']:.4g} | {c['holm_p']:.4g} |")
    text+=['','## Limitations','', '- Single training seed; confidence intervals and paired tests quantify question-level uncertainty, not training-seed variability.','- Scorer v2 was revised on development outputs; numeric answer correctness is not proof of correct reasoning. GSM8K may overlap pretraining.','- Batched greedy inference can differ from batch-one generation. Calibration results are recorded separately; batched throughput is not single-request latency.','- GRPO arms are small exploratory pilots; KL disabled; E3 had a nonfinite gradient with finite saved adapter weights. Length shaping used retokenized lengths and a format bonus.','- Halo native training is separate and not claimed completed.','']
    (report/'REPORT.md').write_text('\n'.join(text),encoding='utf-8'); print('\n'.join(text))
    try:
        import matplotlib.pyplot as plt
        for metric,label in [('accuracy','Answer accuracy'),('mean_tokens','Mean generated tokens'),('correct_per_1000_generated_tokens','Correct answers / 1k tokens')]:
            fig,ax=plt.subplots(figsize=(7,4)); ax.bar(list(metrics),[m[metric] for m in metrics.values()]); ax.set_ylabel(label); ax.set_title('GSM8K held-out test'); fig.tight_layout(); fig.savefig(report/(metric+'.png'),dpi=160); plt.close(fig)
    except ImportError: print('Matplotlib unavailable; CSV and report are complete.')

if __name__=='__main__': main()
