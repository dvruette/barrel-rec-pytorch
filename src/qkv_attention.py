import torch
import torch.nn as nn


class QKVAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_attention_heads: int,
        is_causal: bool = True,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.is_causal = is_causal
        self.attention_dropout = attention_dropout

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None):
        input_dtype = x.dtype
        if ctx is None:
            ctx = x

        queries = self.w_q(x)
        keys = self.w_k(ctx)
        values = self.w_v(ctx)

        queries = queries.view(*queries.shape[:2], self.num_attention_heads, self.d_model // self.num_attention_heads).transpose(1, 2)
        keys = keys.view(*keys.shape[:2], self.num_attention_heads, self.d_model // self.num_attention_heads).transpose(1, 2)
        values = values.view(*values.shape[:2], self.num_attention_heads, self.d_model // self.num_attention_heads).transpose(1, 2)

        # attn_score = torch.einsum("bhnd,bhmd->bhnm", queries, keys)
        attn_score = torch.matmul(queries, keys.transpose(-2, -1))
        attn_score = attn_score / (self.d_model // self.num_attention_heads) ** 0.5

        if self.is_causal:
            mask = torch.triu(torch.ones(attn_score.shape[-2:]), diagonal=1).to(attn_score.device)
            attn_score = attn_score.masked_fill(mask == 1, float("-inf"))

        attn_weights = attn_score.to(torch.float32).softmax(dim=-1).to(input_dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # output = torch.einsum("bhnm,bhmd->bhnd", attn_weights, values)
        output = torch.matmul(attn_weights, values)
        output = output.transpose(1, 2).reshape(*x.shape[:2], self.d_model)
        output = self.w_o(output)

        return output
