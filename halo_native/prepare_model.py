from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import json
root=Path('/workspace/halo-smoke')
model='Qwen/Qwen3-4B-Instruct-2507'
p=root/'model_revision.json'
if p.exists(): rev=json.loads(p.read_text())['revision']
else:
    rev=HfApi().model_info(model).sha
    assert rev
    p.write_text(json.dumps({'model':model,'revision':rev},indent=2))
snapshot_download(model,revision=rev,local_dir=root/'base_model')
