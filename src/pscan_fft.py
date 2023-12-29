import torch

"""
FFT-based implementation of PSCAN by Maxim Zubkov
https://github.com/maximzubkov/fft-scan/blob/main/fft-scan.ipynb
"""

def L_at_X(X):
    N, T, D = X.shape
    dtype = X.dtype
    device = X.device

    X_ = X.transpose(0, 1)
    X_ = torch.cat([X_, torch.zeros(T - 1, N, D, dtype=dtype, device=device)], dim=0)

    L = torch.where(
        (torch.arange(2 * T - 1, device=device) <= T - 1),
        1, 
        0
    )
    L = L.unsqueeze(1).unsqueeze(2)

    output = torch.fft.ifft(
        torch.fft.fft(L, dim=0) * torch.fft.fft(X_, dim=0),
        n=2 * T - 1,
        dim=0
    )
    output = output[:T, :, :].transpose(0, 1)
    return output

def U_at_A(A):
    N, T = A.shape
    dtype = A.dtype
    device = A.device

    A_ = A.transpose(0, 1)
    A_ = torch.cat([A_, torch.zeros(T - 1, N, dtype=dtype, device=device)], dim=0)

    L_no_diag = torch.where(
        (torch.arange(2 * T - 1, device=device) >= 1) & (torch.arange(2 * T - 1, device=device) <= T - 1),
        1, 
        0
    )
    L_no_diag = L_no_diag.unsqueeze(1)
    
    L_no_diag_at_A = torch.fft.ifft(
        torch.fft.fft(L_no_diag, dim=0) * torch.fft.fft(A_, dim=0), 
        n=2 * T - 1,
        dim=0
    )
    # Since we add T - 1 of padding zeros to A_log_T
    output = A_.sum(0).unsqueeze(0) - L_no_diag_at_A
    output = output[:T, :].transpose(0, 1)
    return output

def pscan_fft(A, X):
    N, T, D = X.shape
    dtype = X.dtype
    device = X.device

    # A_log \in [N x T]
    A_log = torch.log(A.to(dtype=torch.cfloat))

    UA = U_at_A(A_log)
    W = UA
    W = W.real
    W_max = W.max()
    e_W = torch.exp(W - W_max)
    e_W = e_W.unsqueeze(-1)

    V = -UA + A_log
    V = V.real
    V_max = V.max()
    e_V = torch.exp(V - V_max)
    e_V = e_V.unsqueeze(-1)
    Y_ = e_V * L_at_X(e_W * X) * (torch.exp(V_max + W_max))

    # After exp we no longer have complex components
    Y_ = Y_.real
    Y_ = torch.cat([torch.zeros(N, 1, D, dtype=dtype, device=device), Y_[:, :-1, :]], dim=1) 
    Y = Y_ + X
    return Y  
