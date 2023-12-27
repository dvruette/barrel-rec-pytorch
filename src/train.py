from argparse import ArgumentParser

import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader

from model import Transformer


def get_dataset(seq_len: int = 256):
    ds = load_dataset("tiny_shakespeare")
    # train split includes vocabulary for other splits
    vocab = set()
    for split in ds.keys():
        for x in ds[split]:
            vocab.update(x["text"])
    vocab = sorted(vocab)

    token_to_id = {token: i for i, token in enumerate(vocab)}

    def encode_and_chunk(batch):
        input_ids, labels = [], []
        for x in batch["text"]:
            tokens = [token_to_id[token] for token in x]
            for i in range(0, len(tokens) - 1, seq_len):
                input_ids.append(tokens[i : min(i + seq_len, len(tokens) - 2)])
                labels.append(tokens[i + 1 : i + seq_len + 1])
        
        return {"input_ids": input_ids, "labels": labels}

    ds = ds.map(encode_and_chunk, batched=True, remove_columns=["text"])
    ds = ds.with_format("torch")
    return ds, vocab

def get_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--mlp_expansion_factor", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_lines", type=int, default=64)
    parser.add_argument("--attn_type", type=str, default="qkv")
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--residual_dropout", type=float, default=0.0)
    parser.add_argument("--ln_eps", type=float, default=1e-8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)

    return parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds, vocab = get_dataset(args.max_seq_len)

    model = Transformer(
        vocab_size=len(vocab),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        mlp_expansion_factor=args.mlp_expansion_factor,
        num_attention_heads=args.num_attention_heads,
        num_layers=args.num_layers,
        num_lines=args.num_lines,
        attention_dropout=args.attention_dropout,
        mlp_dropout=args.mlp_dropout,
        residual_dropout=args.residual_dropout,
        attn_type=args.attn_type,
        ln_eps=args.ln_eps,
    ).to(device)

    train_dl = DataLoader(ds["train"], batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=True)
    val_dl = DataLoader(ds["validation"], batch_size=args.batch_size, pin_memory=True, num_workers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with tqdm.tqdm(total=args.epochs * (len(train_dl) + len(val_dl))) as pbar:
        stats = {}
        for epoch in range(args.epochs):
            stats["epoch"] = epoch

            model.train()
            for batch in train_dl:
                loss = get_loss(model, batch, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats["train_loss"] = loss.item()

                pbar.update(1)
                pbar.set_postfix(stats)

            model.eval()
            with torch.no_grad():
                for batch in val_dl:
                    loss = get_loss(model, batch, device)

                    stats["val_loss"] = loss.item()

                    pbar.update(1)
                    pbar.set_postfix(stats)


if __name__ == "__main__":
    args = parse_args()
    main(args)
