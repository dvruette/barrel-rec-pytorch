#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

# copied and adapted from https://fleuret.org/tmp/mygpt.py

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import pscan


def moving_window(x, dim, win_dim, win_size) -> torch.Tensor:
    size, stride = x.size(), x.stride()
    size = size[:dim] + (size[dim] - win_size + 1,) + size[dim + 1 :]
    size = size[:win_dim] + (win_size,) + size[win_dim:]
    stride = stride[:win_dim] + (stride[dim],) + stride[win_dim:]

    return x.as_strided(size=size, stride=stride)


class Caterpillar(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_keys: int,
        d_values: int,
        num_attention_heads: int,
        caterpillar_length: int,
        caterpillar_height: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_keys = d_keys
        self.d_values = d_values
        self.num_attention_heads = num_attention_heads
        self.caterpillar_length = caterpillar_length
        self.caterpillar_height = caterpillar_height
        self.attention_dropout = attention_dropout

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.caterpillar_length = caterpillar_length
        self.caterpillar_height = caterpillar_height
        self.attention_dropout = attention_dropout

        self.w_G = randw(num_attention_heads, caterpillar_height, d_model)
        self.b_G = nn.Parameter(
            torch.full(
                (num_attention_heads, caterpillar_height), fill_value=-num_attention_heads**0.5, dtype=torch.float32
            )
        )

        self.w_K = randw(num_attention_heads, d_keys, d_model)
        self.w_V = randw(num_attention_heads, d_values, d_model)

        self.w_Q = randw(num_attention_heads, d_keys, d_model)
        self.w_O = randw(d_values * num_attention_heads, d_model)

        self.init_K_rec = randw(caterpillar_height, caterpillar_length, d_keys)
        self.init_V_rec = randw(caterpillar_height, caterpillar_length, d_values)

    def _pscan(self, A, X, Y_init) -> torch.Tensor:
        # A: (bs, cat_h, cat_len, ctx_len // cat_len)
        # X: (bs, cat_h, cat_len, ctx_len // cat_len, dim)
        # Y_init: (bs, cat_h, cat_len, dim)

        s = X.size()
        l, T = s[:3].numel(), s[-2]
        A = A.reshape(l, T)
        X = X.reshape(l, T, -1)
        Y_init = Y_init.reshape(l, -1)

        res = pscan.pscan(A, X, Y_init)  # (l, T, dim)
        Y = res.view(s)

        return Y

    def forward(self, X: torch.Tensor):
        N, T, _ = X.size()

        cat_h = self.caterpillar_height
        cat_len = self.caterpillar_length
        
        # we need to pad to a multiple of cat_len
        T_pad = ((T - 1) // cat_len + 1) * cat_len
        X = F.pad(X, (0, 0, 0, T_pad - T), value=0.0)
        
        t0, t1 = cat_len, T_pad + cat_len  # first cat_len tokens are reserved for the initial state

        # NxExTxD where E is the index in the height (since H is already used for heads)
        rec_K = X.new_zeros(N, cat_h, t1, self.d_keys)
        rec_V = X.new_zeros(N, cat_h, t1, self.d_values)
        rec_K[:, :, t0 - cat_len : t0] = self.init_K_rec[None, :, :, :]
        rec_V[:, :, t0 - cat_len : t0] = self.init_V_rec[None, :, :, :]

        ######################################################################
        # Compute the recurrent state

        G = (
            torch.einsum("ntd,hed->nhet", X, self.w_G) + self.b_G[None, :, :, None]
        ).sigmoid()

        K = torch.einsum("ntc,hdc->nhtd", X, self.w_K)  # (bs, n_heads, ctx_len, d_keys)
        V = torch.einsum("ntc,hdc->nhtd", X, self.w_V)  # (bs, n_heads, ctx_len, d_values)

        A = 1 - G.sum(1).clip(max=1)  # (bs, cat_h, ctx_len)
        gated_K = torch.einsum("nhet,nhtd->netd", G, K)  # (bs, cat_h, ctx_len, d_keys)
        gated_V = torch.einsum("nhet,nhtd->netd", G, V)  # (bs, cat_h, ctx_len, d_values)

        A = A.unflatten(2, (cat_len, -1))  # (bs, cat_h, cat_len, ctx_len // cat_len)
        gated_K = gated_K.unflatten(2, (cat_len, -1))  # (bs, cat_h, cat_len, ctx_len // cat_len, d_keys)
        gated_V = gated_V.unflatten(2, (cat_len, -1))  # (bs, cat_h, cat_len, ctx_len // cat_len, d_values)

        init_K = rec_K[:, :, t0 - cat_len : t0]
        init_V = rec_V[:, :, t0 - cat_len : t0]

        next_K = self._pscan(A, gated_K, init_K)
        next_V = self._pscan(A, gated_V, init_V)

        rec_K[:, :, t0:t1] = next_K.flatten(2, 3)  # (bs, cat_h, ctx_len, d_keys)
        rec_V[:, :, t0:t1] = next_V.flatten(2, 3)  # (bs, cat_h, ctx_len, d_values)

        ######################################################################
        # compute the readout

        Q = torch.einsum("ntc,hdc->nhtd", X, self.w_Q)
        uk = moving_window(
            rec_K[:, :, t0 - cat_len + 1 : t1], dim=2, win_dim=3, win_size=cat_len
        )  # (bs, cat_h, ctx_len, cat_len, d_keys)
        uv = moving_window(
            rec_V[:, :, t0 - cat_len + 1 : t1], dim=2, win_dim=3, win_size=cat_len
        )  # (bs, cat_h, ctx_len, cat_len, d_values)

        attn_scores = torch.einsum(
            "nhtd,netld->nhtel",
            Q,
            uk,
        ) / math.sqrt(self.d_keys)  # (bs, n_heads, ctx_len, cat_h, cat_len)
        # compute softmax over the entire caterpillar hidden state (cat_h * cat_len vectors)
        attn_scores = attn_scores.flatten(3).softmax(dim=3).view(attn_scores.size())
        attn_scores = F.dropout(attn_scores, self.attention_dropout, self.training)

        Y = torch.einsum(
            "nhtel,netld->nthd",  # reducing e and l
            attn_scores,
            uv,
        ).flatten(2)  # (bs, ctx_len, n_heads * d_values)

        output = Y @ self.w_O
        output = output[:, :T]  # remove padding
        return output