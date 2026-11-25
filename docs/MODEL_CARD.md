# Model card — ReasonLite adapters

Base: Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`. E1 LoRA: `runs/e1_sft_v2/adapter`; E2 correctness GRPO: `runs/e2_grpo_correctness_v1/adapter`; E3 efficiency GRPO: `runs/e3_grpo_efficiency_v1/adapter`. Each includes adapter config, safetensors, tokenizer assets and training metadata. E1 is the preselected development efficiency candidate. E0 is not redistributed as a full base checkpoint.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
name='Qwen/Qwen2.5-0.5B-Instruct'
rev='7ae557604adf67be50417f59c2c2f167def9a775'
tok=AutoTokenizer.from_pretrained(name, revision=rev, trust_remote_code=False)
base=AutoModelForCausalLM.from_pretrained(name, revision=rev, torch_dtype=torch.float16, trust_remote_code=False).to('cuda')
model=PeftModel.from_pretrained(base,'runs/e1_sft_v2/adapter').eval()
messages=[{'role':'system','content':'Solve the math problem. Show your reasoning, then end with #### <number> on its own line.'},{'role':'user','content':'What is 12 times 7?'}]
inputs=tok(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),return_tensors='pt',add_special_tokens=False).to('cuda')
with torch.inference_mode():
    y=model.generate(**inputs,max_new_tokens=384,do_sample=False,pad_token_id=tok.pad_token_id)
print(tok.decode(y[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True))
```

Research-only use; outputs can be mathematically wrong, incomplete or misleading. Final E1 accuracy 29.87%, E2/E3 30.25%, base 38.82% using the study's scorer. Not suitable for high-stakes decisions or reliable tutoring without independent verification. Do not claim production readiness. Training uses GSM8K; no evaluation of fairness, robustness outside this benchmark, or safety. Check [third-party terms](THIRD_PARTY.md) before redistribution. No independent model-hosting publication has been performed. The adapter files themselves are included in this private repository package.
