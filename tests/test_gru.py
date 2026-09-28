import torch
from mechcal.models.gru import CausalGRU, GRUConfig

def test_gru_causal_and_session_independent():
    torch.manual_seed(1)
    model=CausalGRU(GRUConfig(hidden_size=16,num_layers=1)).eval()
    x=torch.randint(0,106,(2,25))
    with torch.no_grad():
        y=model(x)
        changed=x.clone(); changed[:,12:]=0
        torch.testing.assert_close(y[:,:12],model(changed)[:,:12])
        torch.testing.assert_close(y[:1],model(x[:1]))
        torch.testing.assert_close(y[:,:12],model(x[:,:12]))
