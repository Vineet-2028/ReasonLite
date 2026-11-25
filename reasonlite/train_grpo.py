"""Low-compute TRL 0.17 GRPO pilot: independent arms from a frozen SFT adapter."""
import argparse, json, math, time, platform
from fractions import Fraction
from pathlib import Path
from .data import load_rows, digest, dump_json, messages
from .scoring import grade, SCORING_VERSION

EXPECTED_PARENT = '5d885ada194f925c2f0ab34a25c503679e47c450b8d32009e775d34c2528360b'
EXPECTED_MANIFEST = '3fcde5df1054b269cc4aa4b755e499848564f62d4532e6acde9379e653aa533c'

def shape(text, gold, tokens, mode, cap=256, budget=128):
    if mode not in ('correctness','efficiency') or tokens < 0 or budget <= 0:
        raise ValueError('Invalid reward configuration')
    # TRL 0.17 callback supplies decoded text, not generation token IDs.
    # Near-cap re-tokenized outputs are treated conservatively by the fallback scorer.
    g = grade(text, Fraction(gold), truncated=tokens >= cap-1)
    value = 0.0
    if g['correct']:
        value = 1.0
        if mode == 'efficiency':
            value -= .15 * min(1., max(0,tokens-budget)/budget)
            value += .05 * g['format_ok']
    return float(value), g

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/prepared'); p.add_argument('--parent',default='runs/e1_sft_v2/adapter'); p.add_argument('--out',required=True); p.add_argument('--mode',choices=['correctness','efficiency'],required=True); p.add_argument('--prompts',type=int,default=64); p.add_argument('--smoke',action='store_true'); a=p.parse_args()
    import torch, transformers, trl, peft
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from peft import PeftModel
    from safetensors.torch import load_file
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Enable one visible CUDA GPU'
    assert trl.__version__=='0.17.0' and transformers.__version__=='4.51.3' and peft.__version__=='0.15.2', 'Use the pinned environment'
    data=Path(a.data); parent=Path(a.parent); manifest,rl=load_rows(data,'rl')
    assert SCORING_VERSION=='final-answer-v2'
    assert digest(data/'manifest.json')==EXPECTED_MANIFEST
    assert digest(parent/'adapter_model.safetensors')==EXPECTED_PARENT
    for split in ('sft','validation'):
        _, other=load_rows(data,split); assert not ({r['id'] for r in rl}&{r['id'] for r in other})
    assert 1<=a.prompts<=len(rl)
    rows=rl[:a.prompts]
    parent_meta=json.loads((parent/'reasonlite_training.json').read_text())
    assert parent_meta['model_revision']==manifest['model_revision'] and parent_meta['data_manifest_sha256']==EXPECTED_MANIFEST
    set_seed(manifest['seed']); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(manifest['model'],revision=manifest['model_revision'],trust_remote_code=False)
    tok.padding_side='left'
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    for row in rows:
        ids=tok.apply_chat_template(messages(row['question']),tokenize=True,add_generation_prompt=True)
        assert len(ids)<=512, 'Prompt would be truncated: '+row['id']
    model=AutoModelForCausalLM.from_pretrained(manifest['model'],revision=manifest['model_revision'],torch_dtype=torch.float32,trust_remote_code=False)
    model=PeftModel.from_pretrained(model,str(parent),is_trainable=True)
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    assert trainable==parent_meta['trainable_parameters'] and 0<trainable<total
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    dataset=Dataset.from_list([{'prompt':messages(r['question']),'gold':r['gold'],'prompt_id':r['id']} for r in rows])
    count={'rollouts':0,'correct':0,'valid':0,'format_ok':0,'tokens':0,'reward_sum':0.0,'groups':0,'nonconstant_reward_groups':0}
    cap=256
    def reward_fn(prompts, completions, gold, prompt_id, **kwargs):
        values=[]; records=[]
        assert len(completions)==len(gold)==len(prompt_id)==len(prompts)
        for completion, target, pid in zip(completions,gold,prompt_id):
            text=completion if isinstance(completion,str) else completion[0]['content']
            n=len(tok(text,add_special_tokens=False)['input_ids'])
            value,g=shape(text,target,n,a.mode,cap)
            values.append(value); records.append({'id':pid,'gold':target,'output':text,'retokenized_tokens':n,'reward':value,**g})
            count['rollouts']+=1; count['correct']+=int(g['correct']); count['valid']+=int(g['valid']); count['format_ok']+=int(g['format_ok']); count['tokens']+=n; count['reward_sum']+=value
        # Single process, batch size == generations; verify grouping explicitly.
        assert len(records)==4 and len({r['id'] for r in records})==1, 'Unexpected rollout grouping'
        count['groups']+=1; count['nonconstant_reward_groups']+=int(max(values)-min(values)>1e-8)
        with (out/'rollouts.jsonl').open('a',encoding='utf-8') as f:
            for r in records: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        return values
    config=dict(per_device_train_batch_size=4,gradient_accumulation_steps=1,num_generations=4,max_prompt_length=512,max_completion_length=cap,
        learning_rate=2e-5,num_train_epochs=1,max_steps=1 if a.smoke else -1,temperature=0.8,top_p=1.0,top_k=None,
        beta=0.0,num_iterations=1,epsilon=0.2,loss_type='bnpo',scale_rewards=True,mask_truncated_completions=False,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},disable_dropout=True,fp16=True,bf16=False,
        max_grad_norm=1.0,lr_scheduler_type='constant',logging_steps=1,save_strategy='steps',save_steps=8,save_total_limit=1,
        report_to='none',seed=manifest['seed'],data_seed=manifest['seed'],shuffle_dataset=True,use_vllm=False,remove_unused_columns=False)
    args=GRPOConfig(output_dir=str(out/'checkpoints'),**config)
    trainer=GRPOTrainer(model=model,reward_funcs=reward_fn,args=args,train_dataset=dataset,processing_class=tok)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); result=trainer.train(); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    assert math.isfinite(result.metrics['train_loss']), 'Non-finite training loss'
    assert trainer.state.global_step==(1 if a.smoke else a.prompts), (trainer.state.global_step,a.prompts)
    adapter=out/'adapter'; trainer.save_model(str(adapter)); tok.save_pretrained(str(adapter))
    weights=load_file(str(adapter/'adapter_model.safetensors')); initial=load_file(str(parent/'adapter_model.safetensors'))
    assert weights.keys()==initial.keys()
    assert all(torch.isfinite(w).all().item() for w in weights.values())
    changed=sum(int((weights[k].float()!=initial[k].float()).sum().item()) for k in weights)
    meta={'model_revision':manifest['model_revision'],'data_manifest_sha256':EXPECTED_MANIFEST,'parent_adapter_sha256':EXPECTED_PARENT,
        'adapter_sha256':digest(adapter/'adapter_model.safetensors'),'rl_split_sha256':manifest['files']['rl']['sha256'],'prompt_ids':[r['id'] for r in rows],
        'mode':a.mode,'smoke':a.smoke,'scoring_version':SCORING_VERSION,'configuration':config,'seed':manifest['seed'],'optimizer_steps':trainer.state.global_step,
        'trainable_parameters':trainable,'total_parameters':total,'changed_adapter_values':changed,'train_seconds':elapsed,
        'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'transformers':transformers.__version__,'trl':trl.__version__,'peft':peft.__version__,'python':platform.python_version(),
        'trainer_metrics':result.metrics,'rollout_counts':count,'reward_length_measure':'decoded-text retokenization (not exact generated token count)','reference_kl':'disabled (beta=0)'}
    dump_json(adapter/'reasonlite_training.json',meta)
    from .logging_utils import sanitize
    logs, issues = sanitize(trainer.state.log_history)
    dump_json(out/'training_log.json',logs)
    dump_json(out/'nonfinite_log_observations.json',issues)
    dump_json(out/'train_metrics.json',meta)
    print(json.dumps(meta,indent=2))
    if not a.smoke and changed==0: print('WARNING: no adapter values changed; investigate zero reward variance or skipped updates before interpreting the run.')

if __name__=='__main__': main()
