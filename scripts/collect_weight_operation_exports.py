"""Validate and unpack the seven completed PACE reward-operation exports."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import tarfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = ROOT / 'outputs/weight_llama_operation_download_20260924'
DEST = ROOT / 'outputs/weight_llama_operation_sweep_20260923'
CODES = ('000', '010', '030', '050', '070', '090', '100')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-shape', action='store_true')
    args = parser.parse_args()
    archives = (ROOT / 'outputs/weight_llama_operation_fixedshape_download_20260924'
                if args.fixed_shape else ARCHIVES)
    dest = (ROOT / 'outputs/weight_llama_operation_fixedshape_20260924/production'
            if args.fixed_shape else DEST)
    script = ('scripts/eval_weight_llama_operations_fixedshape.py'
              if args.fixed_shape else 'scripts/eval_weight_llama_operations.py')
    source = (ROOT / script).read_bytes()
    source_hashes = {sha256(source), sha256(source.replace(b'\r\n', b'\n'))}
    audit = {'job_array': '13530770' if args.fixed_shape else '13497698',
             'split': 'val', 'fixed_shape': args.fixed_shape, 'weights': {}}
    for code in CODES:
        name = 'reward_w' + code
        archive = archives / (name + '.tgz')
        expected = {f'{name}/participant_{i:03d}.npz' for i in range(250)}
        expected.add(f'{name}/COMPLETE_shard0.json')
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            names = []
            for member in members:
                path = PurePosixPath(member.name)
                assert not path.is_absolute() and '..' not in path.parts
                assert path.parts[0] == name
                assert member.isfile() or member.isdir(), member.name
                if member.isfile():
                    names.append(member.name)
            assert len(names) == len(set(names))
            assert set(names) == expected, (code, set(names) ^ expected)
            manifest = json.load(tar.extractfile(f'{name}/COMPLETE_shard0.json'))
            assert manifest['weight'] == code and manifest['split'] == 'val'
            assert manifest['limit'] is None
            assert manifest['shard'] == 0 and manifest['shards'] == 1
            assert manifest['indices'] == list(range(250))
            assert manifest['script_sha256'] in source_hashes
            if args.fixed_shape:
                assert manifest['inference'] == 'uncached_fixed_shape_future_suffix'
            data = ROOT / 'outputs/weight_llama_sft_20260916' / (name + '_val.jsonl')
            assert manifest['data_sha256'] == sha256(data.read_bytes())
            assert manifest['operations'] == [
                'neutral50', 'reserved_placeholder', 'delete_reward_tokens',
                'history_only_rewrite']
            for member in members:
                target = dest / member.name
                if member.isfile() and target.exists():
                    assert sha256(target.read_bytes()) == sha256(tar.extractfile(member).read())
            dest.mkdir(parents=True, exist_ok=True)
            tar.extractall(dest, filter='data')
        audit['weights'][code] = {
            'participants': 250, 'archive_sha256': sha256(archive.read_bytes()),
            'script_sha256': manifest['script_sha256'],
            'data_sha256': manifest['data_sha256'],
            'manifest_and_safe_paths_passed': True}
        print(code, '250 participant exports verified')
    audit['participant_files'] = 1750
    (dest / 'DOWNLOAD_AUDIT.json').write_text(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
