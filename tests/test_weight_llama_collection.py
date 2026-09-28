import numpy as np
import pytest
from pathlib import Path
from collect_weight_llama import probabilities, read_mode
from eval_weight_llama import reward_batches


def test_reward_batches_preserve_intact_and_all_unique_donors():
    batches = reward_batches(7,[8,9,8,7,10,11],4)
    assert batches == [[7,8,9,10],[7,11,7,7]]
    assert reward_batches(7,[7,7],4) == [[7,7,7,7]]
    with pytest.raises(AssertionError): reward_batches(7,[8],1)


def test_conditional_probabilities_do_not_mix_vocabulary_mass():
    p = np.array([.6, .2, .1, .1])
    assert np.allclose(probabilities(np.log(p*.8)), p)
    assert np.allclose(probabilities(np.log(p*.2)), p)
    assert not np.allclose(np.log(p*.8), np.log(p*.2))


def test_collection_requires_exact_participant_alignment(monkeypatch):
    class Archive(dict):
        def __enter__(self): return self
        def __exit__(self, *args): return False
    archive=Archive(index=0,participant='wrong',actions=np.zeros(200,int),
                    donor_indices=np.ones(20,int),intact_logp=np.full((200,4),np.log(.25)))
    monkeypatch.setattr(np,'load',lambda _:archive)
    data=dict(base_participant_id=np.array(['correct']*250),action=np.zeros((250,200),int))
    with pytest.raises(AssertionError):
        read_mode(Path('unused'),'reward_w010','choice',data,np.ones((20,250),int))


def test_local_collection_retains_separate_prefix_baseline(monkeypatch):
    class Archive(dict):
        def __enter__(self): return self
        def __exit__(self, *args): return False
    def load(path):
        i = int(path.stem.split('_')[-1])
        return Archive(index=i, participant=str(i), actions=np.zeros(200,int),
                       donor_indices=np.ones(20,int),
                       intact_logp=np.full((200,4),np.log(.25)),
                       local_intact_logp=np.full((50,4),np.log(.24)),
                       local_logp=np.full((20,50,4),np.log(.23)),
                       target_indices=np.arange(150,200))
    monkeypatch.setattr(np,'load',load)
    data=dict(base_participant_id=np.arange(250).astype(str),action=np.zeros((250,200),int))
    result=read_mode(Path('unused'),'reward_w010','local',data,np.ones((20,250),int))
    assert result['local_intact_logp'].shape == (250,50,4)
    assert result['local_logp'].shape == (20,250,50,4)
    assert not np.allclose(result['intact_logp'][:,150:],result['local_intact_logp'])
