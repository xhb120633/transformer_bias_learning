"""Export only train/val observable records. Never open test or audit arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from mechcal.training.spatial_transcripts import VOCAB, CHOICE_IDS, encode_map, render_map


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='outputs/spatial_pilot_v1')
    p.add_argument('--output', default='data/spatial_neural_v1')
    p.add_argument('--refresh-text-only', action='store_true', help='Repair rendering; verify local token arrays unchanged')
    args = p.parse_args()
    root, dest = Path(args.source), Path(args.output)
    dest.mkdir(parents=True, exist_ok=args.refresh_text_only)
    manifest = {'source': str(root), 'splits': ['train', 'val'], 'test_accessed': False,
                'vocab': VOCAB, 'choice_ids': CHOICE_IDS, 'reward_format': 'Python float repr, exact round trip',
                'sequence_unit': 'one independent map, 20 choices, no subject ID in input',
                'choice_only': 'omit ALL reward values including the initial cue reward; retain cue position',
                'primary_B': 'cumulative donor chosen rewards, retain initial cue; distinct from A choice-only retraining',
                'files': {}}
    ids = {}
    for weight in ('000', '010', '030', '050', '070', '090', '100'):
        condition = f'reward_w{weight}'
        for split in ('train', 'val'):
            src = root / condition / f'{split}_observable.jsonl'
            records = [json.loads(s) for s in src.read_text().splitlines()]
            ids[condition, split] = {r['participant'] for r in records}
            assert len(records) == (200 if split == 'train' else 50)
            for mode in ('full', 'choice_only'):
                seqs, masks, participants, rounds, texts = [], [], [], [], []
                for r in records:
                    assert r['grid_size'] == 5 and len(r['rounds']) == 8
                    for ep in r['rounds']:
                        seq, mask = encode_map(ep, mode)
                        seqs.append(seq); masks.append(mask)
                        participants.append(r['participant']); rounds.append(ep['round'])
                        texts.append({'text': render_map(ep, mode), 'participant': r['participant'], 'round': ep['round']})
                length = max(map(len, seqs))
                tokens = np.zeros((len(seqs), length), dtype=np.int64)
                targets = np.zeros_like(tokens, dtype=bool)
                for i, (seq, mask) in enumerate(zip(seqs, masks)):
                    tokens[i, :len(seq)] = seq; targets[i, :len(seq)] = mask
                stem = f'{condition}_{mode}_{split}'
                if args.refresh_text_only:
                    previous = np.load(dest / f'{stem}.npz', allow_pickle=False)
                    assert np.array_equal(previous['tokens'], tokens)
                    assert np.array_equal(previous['choice_target_mask'], targets)
                    assert np.array_equal(previous['participant'], np.array(participants))
                    assert np.array_equal(previous['round'], np.array(rounds))
                else:
                    np.savez_compressed(dest / f'{stem}.npz', tokens=tokens, choice_target_mask=targets,
                                        participant=np.array(participants), round=np.array(rounds))
                (dest / f'{stem}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in texts), encoding='utf-8')
                manifest['files'][stem] = {'sequences': len(seqs), 'targets': int(targets.sum()),
                    'max_local_tokens': length, 'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                    'jsonl_sha256': hashlib.sha256((dest / f'{stem}.jsonl').read_bytes()).hexdigest(),
                    'npz_sha256': hashlib.sha256((dest / f'{stem}.npz').read_bytes()).hexdigest()}
        assert not ids[condition, 'train'] & ids[condition, 'val']
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(manifest['files']), 'test_accessed': False,
                      'max_local_tokens': max(x['max_local_tokens'] for x in manifest['files'].values())}))


if __name__ == '__main__':
    main()
