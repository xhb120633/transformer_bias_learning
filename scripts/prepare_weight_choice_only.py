"""Create paired seven-weight choice-only text using the original history-only format."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from mechcal.analysis.evaluate_centaur_history_only import history_only_text

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest=dict(test_used=False,reward_information_present=False,format='original history_only_text',files={})
    for name in ('000','010','030','050','070','090','100'):
        for split,n in [('train',1000),('val',250)]:
            source=args.source/f'reward_w{name}_{split}.jsonl'
            records=[json.loads(x) for x in source.read_text().splitlines()];assert len(records)==n
            out=args.output/source.name
            with out.open('w',encoding='utf-8',newline='\n') as handle:
                for record in records:
                    text=history_only_text(record['text'])
                    assert re.findall(r'<<([ABCD])>>',text)==re.findall(r'<<([ABCD])>>',record['text'])
                    assert text.count('<<')==200 and not re.search(r'reward|points|receive',text,re.I)
                    handle.write(json.dumps(dict(record,text=text))+'\n')
            manifest['files'][out.name]=dict(n=n,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('CHOICE_ONLY_EXPORT_VERIFIED: 14 files; 200 unchanged choices/session; no reward text; no test access',flush=True)

if __name__=='__main__':main()
