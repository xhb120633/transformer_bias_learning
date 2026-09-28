"""Freeze observable test payloads and shared donor maps; no model scoring."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from mechcal.data.export_centaur_sft import transcript_to_text
from mechcal.analysis.evaluate_centaur_history_only import history_only_text
from mechcal.training.spatial_transcripts import label, LETTERS
from run_weight_curve_a import derangements

OUT = Path('data/llama_test_20260925')
WEIGHTS = ('000','010','030','050','070','090','100')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    for part in ('task_a/full','task_a/choice','task_b'): (OUT/part).mkdir(parents=True,exist_ok=True)
    vocab=json.loads(Path('outputs/restless_pooled_eps10_1000_20260827/metadata.json').read_text())['token_vocab']
    amaps=derangements(250,20,20260917)
    bmaps=derangements(50,20,81073)
    sources={}
    for w in WEIGHTS:
        name='reward_w'+w
        source=Path('outputs/weight_curve_a_20260915')/name/'test_000.npz'
        d=dict(np.load(source,allow_pickle=False));sources[str(source)]=sha(source)
        assert d['action'].shape==(250,200)
        for split in ('train','val'):
            with np.load(source.parent/f'{split}_000.npz',allow_pickle=False) as old:
                assert not set(d['base_participant_id'].astype(str)) & set(old['base_participant_id'].astype(str))
                assert not set(d['schedule_seed'].tolist()) & set(old['schedule_seed'].tolist())
        rows={'full':[],'choice':[]}
        for i in range(250):
            text=transcript_to_text(d['tokens'][i],vocab)
            assert np.array_equal(['ABCD'.index(a) for a in re.findall(r'<<([ABCD])>>',text)],d['action'][i])
            assert np.array_equal([int(a) for a in re.findall(r'receive (\d+) points',text)],d['reward'][i])
            choice=history_only_text(text)
            assert not re.search(r'reward|points|receive',choice,re.I)
            for mode,t in (('full',text),('choice',choice)):
                rows[mode].append(dict(text=t,participant=str(d['base_participant_id'][i]),condition=name,split='test'))
        for mode in rows:
            (OUT/'task_a'/mode/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows[mode]),encoding='utf-8')
        np.savez_compressed(OUT/'task_a'/(name+'_maps.npz'),participants=d['base_participant_id'],donor_mappings=amaps)
        source=Path('outputs/spatial_pilot_v1')/name/'test_observable.jsonl'
        records=[json.loads(s) for s in source.read_text().splitlines()];sources[str(source)]=sha(source)
        assert len(records)==50
        for split in ('train','val'):
            old=[json.loads(s) for s in (source.parent/(split+'_observable.jsonl')).read_text().splitlines()]
            assert not {str(r['participant']) for r in records} & {str(r['participant']) for r in old}
        actions=np.array([[[LETTERS.index(label(c)) for c in ep['choices']] for ep in r['rounds']] for r in records])
        assert actions.shape==(50,8,20)
        (OUT/'task_b'/(name+'.json')).write_text(json.dumps(dict(records=records,actions=actions.tolist(),donor_maps=bmaps.tolist(),test_used=True,split='test')),encoding='utf-8')
    manifest=dict(split='test',weights=list(WEIGHTS),donor_seeds={'task_a':20260917,'task_b':81073},
        donors=20,scoring={'task_a':'trials151-200','task_b':'maps7-8'},source_hashes=sources,
        files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        no_test_model_selection=True,checkpoints='existing validation-selected best checkpoints; no retraining')
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(files=len(manifest['files']),split='test',participant_overlap=False)))

if __name__=='__main__': main()
