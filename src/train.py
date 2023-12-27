import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import Transformer


def get_shakespeare(seq_len: int = 256):
    ds = load_dataset("tiny_shakespeare")
    # train split includes vocabulary for other splits
    vocab = set()
    for split in ds.keys():
        for x in ds[split]:
            vocab.update(x["text"])
    vocab = ["<pad>"] + sorted(vocab)

    token_to_id = {token: i for i, token in enumerate(vocab)}

    def encode_and_chunk(batch):
        input_ids, labels = [], []
        for x in batch["text"]:
            tokens = [token_to_id[token] for token in x]
            for i in range(0, len(tokens) - 1, seq_len):
                ts = tokens[i : i + seq_len + 1]
                if len(ts) < seq_len + 1:
                    # ts += [token_to_id["<pad>"]] * (seq_len + 1 - len(ts))
                    continue
                input_ids.append(ts[:-1])
                labels.append(ts[1:])

        
        return {"input_ids": input_ids, "labels": labels}

    ds = ds.map(encode_and_chunk, batched=True, remove_columns=["text"])
    ds = ds.with_format("torch")
    return ds, len(vocab)


def get_wikitext(seq_len: int = 256, tokenizer: str = "gpt2"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    tokenizer_id = tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def encode_and_chunk(batch, tokenizer=tokenizer_id, seq_len=seq_len):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        input_ids, labels = [], []
        for x in batch["text"]:
            tokens = tokenizer(x)["input_ids"]
            for i in range(0, len(tokens) - 1, seq_len):
                ts = tokens[i : i + seq_len + 1]
                if len(ts) < seq_len + 1:
                    # ts += [int(tokenizer.pad_token_id)] * (seq_len + 1 - len(ts))
                    continue
                input_ids.append(ts[:-1])
                labels.append(ts[1:])

        
        return {"input_ids": input_ids, "labels": labels}
    
    ds = ds.map(encode_and_chunk, batched=True, remove_columns=["text"], num_proc=min(32, os.cpu_count()))
    ds = ds.with_format("torch")
    return ds, tokenizer.vocab_size

def get_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shakespeare")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--mlp_expansion_factor", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_lines", type=int, default=64)
    parser.add_argument("--attn_type", type=str, default="dumb_rec")
    parser.add_argument("--attention_dropout", type=float, default=0.0)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--residual_dropout", type=float, default=0.0)
    parser.add_argument("--ln_eps", type=float, default=1e-8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=128)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="dvruette")

    return parser.parse_args()

def main(args):
    if args.wandb:
        import wandb

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.dataset == "shakespeare":
        ds, vocab_size = get_shakespeare(args.max_seq_len)
    elif args.dataset == "wikitext":
        ds, vocab_size = get_wikitext(args.max_seq_len)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model = Transformer(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        mlp_expansion_factor=args.mlp_expansion_factor,
        num_attention_heads=args.num_heads,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.wandb:
        wandb.init(entity=args.wandb_entity, project="barrel-rec", config=args)
        wandb.watch(model, log_freq=100)

    global_step = 0
    steps_per_epoch = len(train_dl) + min(args.eval_batches, len(val_dl))
    with tqdm.tqdm(total=args.epochs * steps_per_epoch) as pbar:
        stats = {}
        for epoch in range(args.epochs):
            stats["epoch"] = epoch

            for batch in train_dl:
                model.train()
                loss = get_loss(model, batch, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # plot gradient norm
                norms = []
                for p in model.parameters():
                    if p.grad is not None:
                        norms.append(p.grad.norm().item())

                stats["train_loss"] = 0.9 * stats.get("train_loss", loss.item()) + 0.1 * loss.item()


                if args.wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_loss_ema": stats["train_loss"],
                        "grad_norm": np.mean(norms),
                        "epoch": epoch,
                    }, step=global_step)

                global_step += 1
                pbar.update(1)
                pbar.set_postfix(stats)

                if (global_step + 1) % args.eval_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        losses = []
                        for i, batch in enumerate(val_dl):
                            if i >= args.eval_batches:
                                break
                            loss = get_loss(model, batch, device)

                            losses.append(loss.item())

                            pbar.update(1)
                        stats["val_loss"] = sum(losses) / len(losses)
                        pbar.set_postfix(stats)


                    if args.wandb:
                        wandb.log({
                            "val_loss": stats["val_loss"],
                        }, step=global_step)


if __name__ == "__main__":
    args = parse_args()
    main(args)
