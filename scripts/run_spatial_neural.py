"""Sequential local spatial GRU/Transformer suite; no test evaluation."""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from mechcal.models.gru import CausalGRU, GRUConfig
from mechcal.models.causal_transformer import CausalTransformer, TransformerConfig
from mechcal.training.spatial_transcripts import VOCAB, CHOICE_IDS


def evaluate(model, loader, device):
    model.eval()
    nll = conditional = correct = count = 0
    with torch.no_grad():
        for tokens, mask in loader:
            tokens, mask = tokens.to(device), mask.to(device)
            logits = model(tokens[:, :-1])[mask[:, 1:]].float()
            labels = tokens[:, 1:][mask[:, 1:]]
            targets = labels - CHOICE_IDS[0]
            nll += F.cross_entropy(logits, labels, reduction='sum').item()
            selected = logits[:, CHOICE_IDS]
            conditional += F.cross_entropy(selected, targets, reduction='sum').item()
            correct += (selected.argmax(-1) == targets).sum().item()
            count += labels.numel()
    return {'full_vocab_nll': nll/count, 'choice_nll': conditional/count, 'accuracy': correct/count, 'n_choices': count}


def run(args, family, mode, weight, seed):
    out = Path(args.output) / f'{family}_{mode}_reward_w{weight}_seed{seed}'
    if (out / 'metrics.json').exists():
        return
    if out.exists():
        raise RuntimeError(f'Unfinished output exists: {out}; inspect before resuming')
    out.mkdir(parents=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    data, source_hashes = {}, {}
    import hashlib
    for split in ('train', 'val'):
        path = Path(args.data) / f'reward_w{weight}_{mode}_{split}.npz'
        a = np.load(path, allow_pickle=False)
        data[split] = TensorDataset(torch.from_numpy(a['tokens']), torch.from_numpy(a['choice_target_mask']))
        source_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    length = max(d.tensors[0].shape[1] for d in data.values())
    if family == 'gru':
        cfg = GRUConfig(vocab_size=len(VOCAB), embedding_dim=64, hidden_size=256, num_layers=2, dropout=.1)
        model = CausalGRU(cfg).to(device)
        lr, epochs, patience, delta, effective, warmup = .001, 200, 20, .0001, 64, 32
    else:
        cfg = TransformerConfig(vocab_size=len(VOCAB), max_sequence_length=length,
                                d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=.1)
        model = CausalTransformer(cfg).to(device)
        lr, epochs, patience, delta, effective, warmup = .0003, 60, 8, .0002, 128, 150
    if args.smoke:
        epochs = 1
        data = {k: TensorDataset(*(t[:8] for t in d.tensors)) for k, d in data.items()}
    batch = min(32, effective)
    loaders = {s: DataLoader(d, batch_size=batch, shuffle=(s == 'train'), num_workers=0) for s, d in data.items()}
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
    steps_per_epoch = math.ceil(len(data['train']) / effective)
    total_steps = steps_per_epoch * epochs
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step-warmup) / max(1, total_steps-warmup))
        return .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_factor)
    configuration = {'family': family, 'mode': mode, 'weight': weight, 'seed': seed,
        'model': cfg.to_dict(), 'learning_rate': lr, 'max_epochs': epochs, 'patience': patience,
        'min_delta': delta, 'effective_batch': effective, 'microbatch': batch, 'precision': 'fp32',
        'source_sha256': source_hashes, 'test_accessed': False, 'torch': torch.__version__,
        'loss': 'full vocabulary cross entropy at choice targets only', 'selection': 'validation full_vocab_nll'}
    (out / 'config.json').write_text(json.dumps(configuration, indent=2))
    best, anchor, bad, best_epoch = float('inf'), float('inf'), 0, 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); optim.zero_grad(set_to_none=True)
        epoch_loss = epoch_count = group_count = 0
        for i, (tokens, mask) in enumerate(loaders['train']):
            tokens, mask = tokens.to(device), mask.to(device)
            logits = model(tokens[:, :-1])[mask[:, 1:]]
            labels = tokens[:, 1:][mask[:, 1:]]
            loss = F.cross_entropy(logits, labels, reduction='sum')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            group_count += labels.numel(); epoch_count += labels.numel(); epoch_loss += loss.item()
            if (i + 1) % (effective // batch) == 0 or i + 1 == len(loaders['train']):
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_count)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optim.step(); scheduler.step(); optim.zero_grad(set_to_none=True); group_count = 0
        val = evaluate(model, loaders['val'], device)
        score = val['full_vocab_nll']
        if not math.isfinite(score):
            raise RuntimeError('Nonfinite validation loss')
        if score < best:
            best, best_epoch = score, epoch
            torch.save({'model': model.state_dict(), 'config': configuration, 'epoch': epoch, 'val': val}, out / 'best.pt')
        if score < anchor - delta:
            anchor, bad = score, 0
        else:
            bad += 1
        row = {'epoch': epoch, 'train_nll': epoch_loss/epoch_count, 'val': val,
               'best_epoch': best_epoch, 'elapsed_seconds': time.time()-started}
        with (out / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(out.name, json.dumps(row), flush=True)
        if bad >= patience:
            break
    checkpoint = torch.load(out / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    metrics = {'best_epoch': best_epoch, 'epochs_run': epoch, 'val': evaluate(model, loaders['val'], device),
               'test_accessed': False, 'elapsed_seconds': time.time()-started}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    del model, optim, checkpoint
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/spatial_neural_v1')
    p.add_argument('--output', default='outputs/spatial_neural_v1')
    p.add_argument('--families', nargs='+', default=['gru', 'transformer'])
    p.add_argument('--modes', nargs='+', default=['full', 'choice_only'])
    p.add_argument('--weights', nargs='+', default=['000', '010', '030', '050', '070', '090', '100'])
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    # Finish a first seed across both families before replicating.
    for seed in args.seeds:
        for weight in args.weights:
            for family in args.families:
                for mode in args.modes:
                    run(args, family, mode, weight, seed)
    (Path(args.output) / 'TRAINING_COMPLETE').write_text('All requested runs completed; test not evaluated.\n')


if __name__ == '__main__':
    main()
