import torch
import torch.nn as nn

from pscan import pscan


class DumbRec(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_keys: int,
        d_values: int,
        num_attention_heads: int,
        num_lines: int,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_keys = d_keys
        self.d_values = d_values
        self.num_attention_heads = num_attention_heads
        self.num_lines = num_lines
        self.dropout = attention_dropout

        if not d_values * num_attention_heads == d_model:
            raise ValueError("d_values * num_attention_heads must equal d_model")

        self.keys = nn.Parameter(d_keys ** -0.5 * torch.randn(num_lines, d_keys), requires_grad=True)
        self.v_init = nn.Parameter(torch.zeros(num_lines, d_values), requires_grad=True)

        self.w_q_r = nn.Linear(d_model, d_keys * num_attention_heads, bias=False)
        self.w_q_w = nn.Linear(d_model, d_keys * num_attention_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_values * num_attention_heads, bias=False)
        self.w_o = nn.Linear(d_values * num_attention_heads, d_model, bias=False)

        self.w_q_r.weight.data.normal_(mean=0.0, std=d_keys ** -0.5)
        self.w_q_w.weight.data.normal_(mean=0.0, std=d_keys ** -0.5)
        self.w_v.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        # self.w_o.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        self.w_o.weight.data.copy_(torch.eye(d_model))

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        bs, seq_len, dim = x.shape

        queries_r = self.w_q_r(x)  # [batch_size, seq_len, num_heads * d_keys]
        queries_w = self.w_q_w(x)  # [batch_size, seq_len, num_heads * d_keys]
        values = self.w_v(x)  # [batch_size, seq_len, num_heads * d_values]

        queries_r = queries_r.view(*queries_r.shape[:2], self.num_attention_heads, self.d_keys)
        queries_w = queries_w.view(*queries_w.shape[:2], self.num_attention_heads, self.d_keys)
        values = values.view(*values.shape[:2], self.num_attention_heads, self.d_values)  # [batch_size, seq_len, num_heads, d_values]

        score_r = torch.einsum("blhd,md->blhm", queries_r, self.keys) / (self.d_keys ** 0.5)
        score_w = torch.einsum("blhd,md->blhm", queries_w, self.keys) / (self.d_keys ** 0.5)

        attn_r = score_r.to(torch.float32).softmax(dim=-1).to(input_dtype)  # [batch_size, seq_len, num_heads, num_lines]
        attn_w = score_w.to(torch.float32).softmax(dim=-1).to(input_dtype)  # [batch_size, seq_len, num_heads, num_lines]

        attn_r = nn.functional.dropout(attn_r, p=self.dropout, training=self.training)
        attn_w = nn.functional.dropout(attn_w, p=self.dropout, training=self.training)

        acc_vals = torch.einsum("blhm,blhd->blmd", attn_w, values)  # [batch_size, seq_len, num_lines, d_values]
        alphas = 1 - attn_w.sum(dim=-2)  # [batch_size, seq_len, num_lines]

        acc_vals = acc_vals.transpose(1, 2).reshape(bs * self.num_lines, seq_len, self.d_values)  # [batch_size * num_lines, seq_len, d_values]
        alphas = alphas.transpose(1, 2).reshape(bs * self.num_lines, seq_len)  # [batch_size * num_lines, seq_len]
        v_init = self.v_init.unsqueeze(0).expand(bs, -1, -1).reshape(bs * self.num_lines, self.d_values)  # [batch_size * num_lines, d_values]

        cum_vals = pscan(alphas.to(torch.float64), acc_vals.to(torch.float64), v_init.to(torch.float64))  # [batch_size * num_lines, seq_len, d_values]
        cum_vals = cum_vals.view(-1, self.num_lines, *cum_vals.shape[1:]).to(input_dtype)  # [batch_size, num_lines, seq_len, d_values]

        result = torch.einsum("blhm,bmld->blhd", attn_r, cum_vals)  # [batch_size, seq_len, num_heads, d_values]
        result = result.reshape(*result.shape[:2], -1)  # [batch_size, seq_len, num_heads * d_values = d_model]
        result = self.w_o(result)

        return result
