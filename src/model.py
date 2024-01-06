from typing import Literal

import torch
import torch.nn as nn

from qkv_attention import QKVAttention
from dumb_rec import DumbRec
from barrel_rec import BarrelRec
from caterpillar import Caterpillar


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.w_1 = nn.Linear(d_model, d_model * expansion_factor)
        self.w_2 = nn.Linear(d_model * expansion_factor, d_model)

        # self.w_1.weight.data.normal_(mean=0.0, std=0.02)
        # self.w_2.weight.data.normal_(mean=0.0, std=0.02)
        # self.w_1.bias.data.fill_(0.0)
        # self.w_2.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        output = self.w_1(x)
        output = nn.functional.gelu(output)
        output = nn.functional.dropout(output, p=self.dropout, training=self.training)
        output = self.w_2(output)

        return output
    

class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_expansion_factor: int,
        num_attention_heads: int,
        num_lines: int = 64,  # only used for dumb_rec
        caterpillar_length: int = 8,  # only used for caterpillar
        caterpillar_height: int = 64,  # only used for caterpillar
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        is_causal: bool = True,
        attn_type: Literal["qkv", "dumb_rec", "barrel_rec", "caterpillar"] = "qkv",
        ln_eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_expansion_factor = mlp_expansion_factor
        self.num_attention_heads = num_attention_heads
        self.num_lines = num_lines
        self.attention_dropout = attention_dropout
        self.mlp_dropout = mlp_dropout
        self.residual_dropout = residual_dropout
        self.attn_type = attn_type
        self.ln_eps = ln_eps

        if attn_type == "qkv":
            self.attention = QKVAttention(d_model, num_attention_heads, attention_dropout=attention_dropout, is_causal=is_causal)
        elif attn_type == "dumb_rec":
            if not is_causal:
                raise ValueError("DumbRec only supports causal attention")
            self.attention = DumbRec(
                d_model=d_model,
                d_keys=d_model // num_attention_heads,
                d_values=d_model // num_attention_heads,
                num_attention_heads=num_attention_heads,
                num_lines=num_lines,
                attention_dropout=attention_dropout,
            )
        elif attn_type == "barrel_rec":
            if not is_causal:
                raise ValueError("BarrelRec only supports causal attention")
            self.attention = BarrelRec(
                d_model=d_model,
                d_keys=d_model // num_attention_heads,
                d_values=d_model // num_attention_heads,
                num_attention_heads=num_attention_heads,
                num_lines=num_lines,
                attention_dropout=attention_dropout,
            )
        elif attn_type == "caterpillar":
            if not is_causal:
                raise ValueError("Caterpillar only supports causal attention")
            self.attention = Caterpillar(
                d_model=d_model,
                d_keys=d_model // num_attention_heads,
                d_values=d_model // num_attention_heads,
                num_attention_heads=num_attention_heads,
                caterpillar_height=caterpillar_length,
                caterpillar_length=caterpillar_height,
                attention_dropout=attention_dropout,
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_type}")
        
        self.pre_norm = nn.LayerNorm(d_model, eps=ln_eps)
        self.post_norm = nn.LayerNorm(d_model, eps=ln_eps)
        self.mlp = MLP(d_model, expansion_factor=mlp_expansion_factor, dropout=mlp_dropout)

        # self.pre_norm.weight.data.fill_(1.0)
        # self.pre_norm.bias.data.fill_(0.0)
        # self.post_norm.weight.data.fill_(1.0)
        # self.post_norm.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor = None):
        residual = x
        x = self.pre_norm(x)
        if self.attn_type == "qkv":
            x = self.attention(x, ctx=ctx)
        else:
            x = self.attention(x)

        x = nn.functional.dropout(x, p=self.residual_dropout, training=self.training)
        x = x + residual

        residual = x
        x = self.post_norm(x)
        x = self.mlp(x)
        x = nn.functional.dropout(x, p=self.residual_dropout, training=self.training)
        x = x + residual

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 512,
        mlp_expansion_factor: int = 4,
        num_attention_heads: int = 8,
        num_layers: int = 6,
        num_lines: int = 64,  # only used for dumb_rec
        caterpillar_length: int = 8,  # only used for caterpillar
        caterpillar_height: int = 64,  # only used for caterpillar
        attention_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        is_causal: bool = True,
        attn_type: Literal["qkv", "dumb_rec", "barrel_rec", "caterpillar"] = "qkv",
        ln_eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.mlp_expansion_factor = mlp_expansion_factor
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_lines = num_lines
        self.attention_dropout = attention_dropout
        self.mlp_dropout = mlp_dropout
        self.residual_dropout = residual_dropout
        self.is_causal = is_causal
        self.attn_type = attn_type
        self.ln_eps = ln_eps

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(max_seq_len, d_model), requires_grad=True)  # learned absolute positional embedding

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                mlp_expansion_factor=mlp_expansion_factor,
                num_attention_heads=num_attention_heads,
                num_lines=num_lines,
                caterpillar_length=caterpillar_length,
                caterpillar_height=caterpillar_height,
                attention_dropout=attention_dropout,
                mlp_dropout=mlp_dropout,
                residual_dropout=residual_dropout,
                is_causal=is_causal,
                attn_type=attn_type,
                ln_eps=ln_eps,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model, eps=ln_eps)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

        self.init_weights()

    def init_weights(self):
        init_range = 0.02
        self.embedding.weight.data.normal_(mean=0.0, std=init_range)
        self.pos_embedding.data.normal_(mean=0.0, std=init_range)

        if self.attn_type == "qkv":
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=init_range)
                    if module.bias is not None:
                        module.bias.data.fill_(0.0)
                    if "w_o" in name:
                        # module.weight.data.copy_(torch.eye(self.d_model))
                        module.weight.data.normal_(mean=0.0, std=init_range / (2*self.num_layers) ** 0.5)
                if isinstance(module, nn.LayerNorm):
                    module.weight.data.fill_(1.0)
                    module.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = x + self.pos_embedding[:x.size(1)]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.out(x)

        return x
