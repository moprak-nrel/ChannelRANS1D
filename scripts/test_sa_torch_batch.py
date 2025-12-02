import torch
from channelrans1d.sa_torch import SpalartAllmarasTorch

def test_batched():
    ny = 4
    M  = 3
    N  = 2

    states = torch.randn(M, 8 * ny)
    params = torch.randn(N, 5)

    states_mn = states.unsqueeze(1).expand(M, N, 8*ny).reshape(M*N, 8*ny)
    params_mn = params.unsqueeze(0).expand(M, N, 5).reshape(M*N, 5)
    model_mn = SpalartAllmarasTorch(Re_tau_round=5200, params=params_mn)
    model_mn.ny = ny
    model_mn.Y = torch.arange(1, ny + 1, dtype=torch.float32)
    out_mn = model_mn.get_dXdt(states_mn)

    for idx in range(M*N):
        i = idx // N
        j = idx % N
        model = SpalartAllmarasTorch(Re_tau_round=5200, params=params[j])
        model.ny = ny
        model.Y = torch.arange(1, ny + 1, dtype=torch.float32)
        out_ind = model.get_dXdt(states[i])
        assert torch.allclose(out_mn[idx], out_ind, atol=1e-6), f"Mismatch at state {i}, param {j}"

if __name__ == "__main__":
    test_batched()
