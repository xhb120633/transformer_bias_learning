"""Verify frozen payloads and record all reused adapter hashes before test jobs."""
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    base=Path('data/llama_test_20260925')
    manifest=json.loads((base/'MANIFEST.json').read_text())
    for relative,expected in manifest['files'].items():
        assert sha(base/relative)==expected,relative
    adapters={}
    for w in ('000','010','030','050','070','090','100'):
        for group in ('weight_llama_sft_20260916','weight_choice_only_sft_20260916'):
            folder=Path('outputs')/group/f'reward_w{w}_seed100'
            adapters[str(folder)]={f:sha(folder/f) for f in ('adapter_model.safetensors','adapter_config.json')}
        for mode in ('full','choice_only'):
            folder=Path('outputs/spatial_llama_v1')/f'reward_w{w}_{mode}_seed100'
            adapters[str(folder)]={f:sha(folder/f) for f in ('adapter_model.safetensors','adapter_config.json')}
    scripts={str(p):sha(p) for p in [Path('scripts/eval_weight_llama_test_20260925.py'),
        Path('scripts/eval_spatial_llama_test_20260925.py'),Path('scripts/slurm/eval_llama_test_main_20260925.sh'),
        Path('src/mechcal/training/spatial_transcripts.py'),Path('src/mechcal/analysis/evaluate_centaur_reward_placeholder.py')]}
    out=Path('outputs/llama_test_20260925');out.mkdir(exist_ok=True)
    target=out/'FROZEN_INPUTS.json'
    value=dict(split='test',adapters=adapters,scripts=scripts,payload_manifest_sha256=sha(base/'MANIFEST.json'))
    if target.exists():assert json.loads(target.read_text())==value
    else:target.write_text(json.dumps(value,indent=2))
    print(json.dumps(dict(adapters=len(adapters),payloads=len(manifest['files']),frozen_inputs_sha256=sha(target))))

if __name__=='__main__':main()
