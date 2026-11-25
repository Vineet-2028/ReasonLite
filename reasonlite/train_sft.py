"""Single-GPU completion-only LoRA SFT; no test split is loaded."""
import argparse
import json
import time
from pathlib import Path
from .data import load_rows, messages, digest, dump_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', default='data/prepared'); p.add_argument('--out', required=True)
    p.add_argument('--config', default='configs/sft.json'); p.add_argument('--max-steps', type=int, default=-1)
    a = p.parse_args(); config = json.loads(Path(a.config).read_text()); manifest, rows = load_rows(a.data,'sft')
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig
    if not torch.cuda.is_available(): raise RuntimeError('SFT requires a CUDA GPU for this recipe')
    if torch.cuda.device_count() != 1: raise RuntimeError('Use CUDA_VISIBLE_DEVICES=0 for this single-GPU recipe')
    set_seed(manifest['seed']); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(manifest['model'], revision=manifest['model_revision'], trust_remote_code=False)
    bf16 = torch.cuda.is_bf16_supported()
    # Float32 weights on T4 with AMP avoids FP16 trainable-gradient unscale errors.
    model = AutoModelForCausalLM.from_pretrained(manifest['model'], revision=manifest['model_revision'],
                      torch_dtype=torch.bfloat16 if bf16 else torch.float32, trust_remote_code=False)
    model.config.use_cache = False
    examples = []; skipped = []
    for r in rows:
        prompt = messages(r['question']); completion = [{'role':'assistant','content':r['solution']}]
        ids = tok.apply_chat_template(prompt+completion, tokenize=True, add_generation_prompt=False)
        if len(ids) > config['max_length']: skipped.append(r['id']); continue
        examples.append({'prompt':prompt,'completion':completion})
    if not examples: raise RuntimeError('All training examples exceed max_length')
    lora = LoraConfig(r=config['rank'], lora_alpha=config['alpha'], lora_dropout=config['dropout'],
                      target_modules=config['target_modules'], task_type='CAUSAL_LM', bias='none')
    args = SFTConfig(output_dir=str(out/'checkpoints'), max_length=config['max_length'], completion_only_loss=True,
          eos_token='<|im_end|>', packing=False, per_device_train_batch_size=config['batch_size'],
          gradient_accumulation_steps=config['gradient_accumulation'], num_train_epochs=config['epochs'],
          max_steps=a.max_steps, learning_rate=config['learning_rate'], warmup_ratio=0.05, lr_scheduler_type='cosine',
          logging_steps=1, save_strategy='epoch', save_total_limit=1, report_to='none', seed=manifest['seed'],
          bf16=bf16, fp16=not bf16, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False})
    trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(examples), processing_class=tok, peft_config=lora)
    # Fail fast if the installed trainer isn't masking the prompt.
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch['labels'][0]; attention = batch['attention_mask'][0].bool()
    if not ((labels[attention] == -100).any() and (labels[attention] != -100).any()):
        raise RuntimeError('Completion-only loss mask is not present')
    trainable = sum(x.numel() for x in trainer.model.parameters() if x.requires_grad)
    total = sum(x.numel() for x in trainer.model.parameters())
    torch.cuda.reset_peak_memory_stats(); t = time.perf_counter(); result = trainer.train(); torch.cuda.synchronize()
    elapsed = time.perf_counter()-t
    adapter = out/'adapter'; trainer.save_model(str(adapter)); tok.save_pretrained(str(adapter))
    meta = {'model_revision':manifest['model_revision'],'data_manifest_sha256':digest(Path(a.data)/'manifest.json'),
            'sft_split_sha256':manifest['files']['sft']['sha256'],'configuration':config,'max_steps':a.max_steps,
            'examples_used':len(examples),'skipped_ids':skipped,'trainable_parameters':trainable,'total_parameters':total,
            'train_seconds':elapsed,'peak_allocated_gpu_mib':torch.cuda.max_memory_allocated()/2**20,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'precision':'bf16' if bf16 else 'fp16_amp',
            'trainer_metrics':result.metrics}
    dump_json(adapter/'reasonlite_training.json',meta); dump_json(out/'train_metrics.json',meta)
    dump_json(out/'training_log.json',trainer.state.log_history); print(json.dumps(meta,indent=2))

if __name__ == '__main__': main()
