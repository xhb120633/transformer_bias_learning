"""Uncached batched causal choice scoring, validation maps only."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
from mechcal.training.spatial_transcripts import render_map, LETTERS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weight',required=True)
    ap.add_argument('--mode',choices=['full','choice_only'],required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--test-main',action='store_true')
    args=ap.parse_args()
    import unsloth
    import torch
    from unsloth import FastLanguageModel
    from trl import DataCollatorForCompletionOnlyLM
    name='reward_w'+args.weight
    source=Path('data/spatial_llama_eval_v1')/(name+'.json')
    if args.test_main: source=Path('data/llama_test_20260925/task_b')/(name+'.json')
    data=json.loads(source.read_text()); records=data['records']; maps=np.array(data['donor_maps'])
    adapter=Path('outputs/spatial_llama_v1')/(name+'_'+args.mode+'_seed100')
    out=Path('outputs/spatial_llama_eval_fixedshape_v2')/('smoke' if args.smoke else 'production')/name/args.mode
    if args.test_main: out=Path('outputs/llama_test_20260925/task_b')/('smoke' if args.smoke else 'production')/name/args.mode
    out.mkdir(parents=True,exist_ok=True)
    model,tok=FastLanguageModel.from_pretrained(model_name=str(adapter),max_seq_length=32768,dtype=None,load_in_4bit=True)
    FastLanguageModel.for_inference(model); model.eval()
    tok.pad_token_id=0; tok.padding_side='right'
    collator=DataCollatorForCompletionOnlyLM(response_template=tok(' <<').input_ids[1:],instruction_template=tok('>>').input_ids[1:],tokenizer=tok)
    choices=[tok.encode(c,add_special_tokens=False) for c in LETTERS]
    assert all(len(c)==1 for c in choices)
    choices=[c[0] for c in choices]; device=next(model.parameters()).device

    def encode(text):
        enc=tok(text,truncation=False)
        b=collator([enc]); pos=b['labels'][0].ne(-100).nonzero().flatten()
        acts=np.array([LETTERS.index(c) for c in re.findall(r'<<([A-Y])>>',text)])
        assert len(pos)==20 and np.array_equal(b['input_ids'][0,pos].numpy(),np.array(choices)[acts])
        return b['input_ids'][0],pos

    @torch.inference_mode()
    def forward(batch):
        # Identical shapes for every intact/donor call, including smoke checks.
        # This controls quantized-kernel rounding driven by padding length.
        length=1024
        assert len(batch)<=args.batch_size and all(len(x[0])<=length for x in batch)
        padded=batch+[batch[0]]*(args.batch_size-len(batch))
        ids=torch.zeros((args.batch_size,length),dtype=torch.long,device=device)
        mask=torch.zeros_like(ids)
        for i,(tokens,pos) in enumerate(padded): ids[i,:len(tokens)]=tokens.to(device); mask[i,:len(tokens)]=1
        result=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
        lp=torch.stack([result.logits[i,(pos-1).to(device)].float().log_softmax(-1)[:,choices] for i,(_,pos) in enumerate(batch)]).cpu().numpy()
        assert np.isfinite(lp).all() and np.all(np.exp(lp).sum(-1)<=1.0001)
        return lp

    # Same-shape future-token perturbation must not affect the first prediction.
    probe=encode(render_map(records[0]['rounds'][0],args.mode))
    changed=probe[0].clone(); changed[probe[1][0]:]=choices[0]
    checks=forward([probe,(changed,probe[1])])
    error=float(np.max(np.abs(checks[0,0]-checks[1,0])))
    assert error < .003, ('causal prefix check',error)
    start=time.time()
    for i in range(2 if args.smoke else 50):
        target=out/f'participant_{i:03d}.npz'
        if target.exists(): continue
        episodes=records[i]['rounds']
        texts=[render_map(ep,args.mode) for ep in episodes]
        repeats=2 if args.smoke else 20
        if args.mode=='full':
            for r in range(repeats):
                for j in (6,7):
                    ep=dict(episodes[j]); ep['rewards']=records[int(maps[r,i])]['rounds'][j]['rewards']
                    texts.append(render_map(ep,args.mode))
        encoded=[encode(t) for t in texts]
        values=np.concatenate([forward(encoded[k:k+args.batch_size]) for k in range(0,len(encoded),args.batch_size)])
        payload=dict(index=i,participant=str(records[i]['participant']),actions=np.array(data['actions'][i]),donor_indices=maps[:repeats,i],intact_logp=values[:8])
        if args.mode=='full':
            payload['donor_logp']=values[8:].reshape(repeats,2,20,25)
            first_error=float(np.max(np.abs(payload['donor_logp'][:,:,0]-values[None,6:8,0])))
            assert first_error < .003, ('actual donor unchanged-prefix check',i,first_error)
            payload['first_choice_max_logp_difference']=first_error
        temporary=target.with_suffix('.partial.npz')
        np.savez_compressed(temporary,**payload)
        temporary.replace(target)
        print(f'COMPLETE participant={i} elapsed={time.time()-start:.1f}',flush=True)
    manifest=dict(complete=True,smoke=args.smoke,test_used=args.test_main,split='test' if args.test_main else 'val',weight=args.weight,mode=args.mode,causal_max_difference=error,batch_size=args.batch_size,fixed_length=1024,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_config_sha256=hashlib.sha256((adapter/'adapter_config.json').read_bytes()).hexdigest())
    (out/'COMPLETE.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
