"""Validation-only observable payload and existing matched donor permutations."""
import json
from pathlib import Path
import numpy as np
from mechcal.training.spatial_transcripts import render_map, label, LETTERS

def main():
    out = Path('data/spatial_llama_eval_v1'); out.mkdir(exist_ok=True)
    for w in ('000','010','030','050','070','090','100'):
        name = 'reward_w'+w
        records = [json.loads(s) for s in (Path('outputs/spatial_pilot_v1')/name/'val_observable.jsonl').read_text().splitlines()]
        z = np.load(Path('outputs/spatial_neural_analysis_v1')/(name+'_matched.npz'))
        assert np.array_equal(z['participants'].astype(str), [str(r['participant']) for r in records])
        actions = np.array([[[LETTERS.index(label(c)) for c in ep['choices']] for ep in r['rounds']] for r in records])
        assert np.array_equal(actions, z['actions'])
        maps = z['donor_maps']
        assert maps.shape == (20,50)
        assert all(np.array_equal(np.sort(m),np.arange(50)) and np.all(m != np.arange(50)) for m in maps)
        for mode in ('full','choice_only'):
            original = [json.loads(s) for s in Path(f'data/spatial_neural_v1/{name}_{mode}_val.jsonl').read_text().splitlines()]
            assert [x['text'] for x in original] == [render_map(ep,mode) for r in records for ep in r['rounds']]
        (out/(name+'.json')).write_text(json.dumps(dict(records=records, donor_maps=maps.tolist(), actions=actions.tolist(), test_used=False)))
    print('PASS: 7 observable-only payloads; exact transcript, participant, action and donor alignment')

if __name__ == '__main__': main()
