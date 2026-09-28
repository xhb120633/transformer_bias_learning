"""Matched trained Transformer torso / reset-head control; no teacher selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
import run_neural_generator_sweep as sweep
import run_random_generator_controls as shared

ROOT=Path('outputs/trained_transformer_generator_20260921')
SOURCE=Path('outputs/weight_neural_full_20260916/transformer_reward_w100_full_seed11/best.pt')

def generator(seed):
    ck=torch.load(SOURCE,map_location='cpu',weights_only=True)
    model=sweep.CausalTransformer(sweep.TransformerConfig(**ck['model_config']))
    model.load_state_dict(ck['model_state'])
    torch.manual_seed(seed)
    # Untie the readout before reset, preserving the learned input embedding.
    model.output=torch.nn.Linear(model.config.d_model,model.config.vocab_size,bias=False)
    assert model.output.weight.data_ptr()!=model.token_embedding.weight.data_ptr()
    for name,value in model.state_dict().items():
        if not name.startswith('output.'):
            assert torch.equal(value,ck['model_state'][name]),name
    assert any(not torch.equal(v,ck['model_state'][k]) for k,v in model.state_dict().items() if k.startswith('output.'))
    for p in model.parameters():p.requires_grad_(False)
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/f'generator_seed{seed}.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,old['model_state'][k]) for k,v in model.state_dict().items())
    else:
        torch.save(dict(model_state=model.state_dict(),model_config=ck['model_config'],head_seed=seed,source=str(SOURCE)),path)
    return model.cuda().eval(),ck['model_config']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    ROOT.mkdir(parents=True,exist_ok=True)
    shared.FAMILY='transformer';shared.generator=generator
    # Reuse the audited causal Transformer generator, without touching random-control outputs.
    shared.ROOT=ROOT/'smoke_checks'
    (shared.ROOT/'transformer').mkdir(parents=True,exist_ok=True)
    shared.smoke('transformer')
    if args.smoke_only:return
    sweep.OUT=ROOT;sweep.SOURCE=SOURCE;sweep.generator=generator;sweep.generate=shared.generate
    sweep.PROTOCOL_EXTRA=dict(generator_family='transformer',
        initialization='Retain trained reward_w100 seed11 torso; untie output from token embedding; fresh bias-free Linear default reset with seeds101/102/103; freeze all parameters',
        policy='w * frozen trained-torso/reset-head policy(real rewards) + (1-w) * same policy(rewards50); causal prefix; own closed-loop choices',
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        shared_generation_sha256=hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        limitations='Inherited symbolic-data-trained torso; one student seed; three head seeds; exploratory validation',
        selection='All21 cells retained; no head selection, temperature adjustment or posthoc rescaling')
    sys.argv=[sys.argv[0]];sweep.main()

if __name__=='__main__':main()
