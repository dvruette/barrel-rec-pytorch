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
from data import get_shakespeare, get_wikitext


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=128)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="dvruette")

    return parser.parse_args()


def get_loss(model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    logits = model(input_ids)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def generate(model, tokenizer, device, max_seq_len=256, max_new_tokens=256, batch_size=1, do_sample=True):
    model.eval()
    with torch.no_grad():
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device).fill_(tokenizer.bos_token_id)
        for i in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :]
            if do_sample:
                logits = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(logits, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if (input_ids == tokenizer.eos_token_id).any(dim=-1).all():
                break

        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        return texts

def main(args):
    if args.wandb:
        import wandb

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "shakespeare":
        ds, vocab_size = get_shakespeare(args.max_seq_len, tokenizer=tokenizer.name_or_path)
    elif args.dataset == "wikitext":
        ds, vocab_size = get_wikitext(args.max_seq_len, tokenizer=tokenizer.name_or_path)
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

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

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

                        samples = generate(
                            model,
                            tokenizer=tokenizer,
                            device=device,
                            max_seq_len=args.max_seq_len,
                            max_new_tokens=128,
                            batch_size=4,
                            do_sample=True,
                        )

                        gen_samples = wandb.Table(columns=["step", "loss", "sample_id", "text"])
                        tqdm.tqdm.write(f"------------ {global_step=} ({datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}) ------------")
                        for i, sample in enumerate(samples):
                            tqdm.tqdm.write(f"[{i}] {sample}")
                            gen_samples.add_data(global_step, stats["val_loss"], i, sample)
                        wandb.log({"samples": gen_samples}, step=global_step)


                    if args.wandb:
                        wandb.log({
                            "val_loss": stats["val_loss"],
                        }, step=global_step)

if __name__ == "__main__":
    args = parse_args()
    main(args)
