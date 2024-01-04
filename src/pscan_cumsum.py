import torch

"""
CumSum-based implementation of PSCAN by Maxim Zubkov
https://github.com/maximzubkov/fft-scan/src/cumsum.py
"""

def pscan_cumsum(A, X):
    N, T, D = X.shape
    device = X.device

    # A_log \in [N x T]
    A_log = torch.log(A.to(dtype=torch.cfloat))

    CS = A_log.cumsum(dim=-1)
    # A_log.sum(dim=-1) + A_log - A_log.cumsum(dim=-1) = A_log[:, ::-1].cumsum(dim=-1)[:, ::-1]
    UA = CS[:, -1].unsqueeze(-1) + A_log - CS

    W = UA
    W_max = W.real.max()
    e_W = torch.exp(W - W_max).real
    e_W = e_W.unsqueeze(-1)

    V = -UA + A_log
    V_max = V.real.max()
    e_V = torch.exp(V - V_max).real
    e_V = e_V.unsqueeze(-1)
    Y_ = e_V * torch.cumsum(e_W * X, dim=1) * (torch.exp(V_max + W_max))

    # After exp we no longer have complex components
    Y_ = torch.cat([torch.zeros(N, 1, D, device=device), Y_[:, :-1, :]], dim=1)
    Y = Y_ + X
    return Y
