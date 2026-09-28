"""Export only training/development seven-weight transcripts; preserve raw order."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from mechcal.data.export_centaur_sft import transcript_to_text
from run_weight_curve_a import WEIGHTS

SOURCE=Path('outputs/weight_curve_a_20260915')
OUT=Path('outputs/weight_llama_sft_20260916')

def main():
    OUT.mkdir(exist_ok=False)
    vocab=json.loads(Path('outputs/restless_pooled_eps10_1000_20260827/metadata.json').read_text())['token_vocab']
    manifest=dict(test_used=False,source=str(SOURCE),seed=100,files={},
        train_n=1000,val_n=250,trials=200,choice_only_loss=True,weight_in_prompt=False)
    for w in WEIGHTS:
        name=f'reward_w{round(w*100):03d}'
        for split,n in [('train',1000),('val',250)]:
            source=SOURCE/name/f'{split}_000.npz';d=dict(np.load(source))
            assert d['action'].shape==(n,200)
            path=OUT/f'{name}_{split}.jsonl'
            with path.open('w',encoding='utf-8',newline='\n') as handle:
                for i in range(n):
                    text=transcript_to_text(d['tokens'][i],vocab)
                    assert np.array_equal(['ABCD'.index(a) for a in re.findall(r'<<([ABCD])>>',text)],d['action'][i])
                    assert np.array_equal([int(a) for a in re.findall(r'receive (\d+) points',text)],d['reward'][i])
                    handle.write(json.dumps(dict(text=text,participant=str(d['base_participant_id'][i]),condition=name,split=split))+'\n')
            manifest['files'][path.name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),n=n)
        maps=np.load(SOURCE/name/'validation_assay.npz')['donor_mappings']
        np.savez_compressed(OUT/f'{name}_val_maps.npz',participants=d['base_participant_id'],donor_mappings=maps)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Exported and verified 14 JSONL files; test not accessed.',flush=True)

if __name__=='__main__':main()
