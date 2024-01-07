#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import torch

######################################################################


class PScan(torch.autograd.Function):
    # Given A is NxTx1 and X is NxTxD, expands A and X in place in O(T),
    # and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))
            A[:, -1].mul_(A[:, -2])

    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0]#.clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        A_star = A.unsqueeze(-1).clone()
        X_star = X.clone()
        PScan.expand_(A_star, X_star)

        ctx.save_for_backward(A_star, X_star, Y_init)
        return A_star * Y_init.unsqueeze(1) + X_star

    @staticmethod
    def backward(ctx, grad_output):
        A_star, X_star, Y_init = ctx.saved_tensors
        grad_A, grad_X, grad_Y_init = None, None, None

        R = grad_output.clone()
        PScan.acc_rev_(A_star, R)
        
        if ctx.needs_input_grad[0]:
            Q = Y_init.unsqueeze(1).expand_as(X_star).clone()
            Q[:, 1:].mul_(A_star[:, :-1]).add_(X_star[:, :-1])
            grad_A = (Q * R).sum(-1)
        if ctx.needs_input_grad[1]:
            grad_X = R
        if ctx.needs_input_grad[2]:
            grad_Y_init = (grad_output * A_star).sum(dim=1)

        return grad_A, grad_X, grad_Y_init


pscan = PScan.apply

######################################################################

if __name__ == "__main__":
    import time, sys

    N, T, D = 2, 1047, 3

    A = torch.rand(N, T, dtype=torch.float64).requires_grad_()
    X = torch.randn(N, T, D, dtype=torch.float64).requires_grad_()
    Y_init = torch.randn(N, D, dtype=torch.float64).requires_grad_()

    # Iterative implementation

    y = Y_init
    s = 0

    for k in range(A.size(1)):
        y = A[:, k, None] * y + X[:, k]
        s = s + y

    s = s.sum()

    gA_ref, gX_ref, gY_init_ref = torch.autograd.grad(
        s, (A, X, Y_init), retain_graph=True
    )

    # parallel scan

    start_time = time.perf_counter()
    for _ in range(1000):
        Y = pscan(A, X, Y_init)
    duration = time.perf_counter() - start_time
    print(f"duration {duration}")

    s = Y.sum()

    gA, gX, gY_init = torch.autograd.grad(s, (A, X, Y_init), retain_graph=True)

    # print(gA)
    # print(gX)
    # print(gY_init)

    print((gA - gA_ref).norm())
    print((gX - gX_ref).norm())
    print((gY_init - gY_init_ref).norm())

    Y1 = pscan(A[:, : T // 2], X[:, : T // 2], Y_init)
    Y2 = pscan(A[:, T // 2 :], X[:, T // 2 :], Y1[:, -1])

    print((Y - torch.cat([Y1, Y2], dim=1)).norm())