import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

from model import Transformer
from data import get_shakespeare, get_wikitext, get_repetition_task, get_slim_pajama, get_grid_task


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="slim_pajama")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--mlp_expansion_factor", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_lines", type=int, default=128)
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
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=128)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default="dvruette")
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def parse_dtype(dtype):
    if dtype == "fp32":
        return torch.float32
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_loss(model_type, model, batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    if model_type == "hf":
        output = model(input_ids=input_ids)
        logits = output.logits
    else:
        logits = model(input_ids)

    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss


def generate(model_type, model, tokenizer, device, max_seq_len=256, max_new_tokens=256, batch_size=1, do_sample=True):
    if model_type == "hf":
        return generate_hf(model, tokenizer, device, max_seq_len, max_new_tokens, batch_size, do_sample)
    else:
        return generate_custom(model, tokenizer, device, max_seq_len, max_new_tokens, batch_size, do_sample)

def generate_custom(model, tokenizer, device, max_seq_len=256, max_new_tokens=256, batch_size=1, do_sample=True):
    model.eval()
    with torch.no_grad():
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device).fill_(tokenizer.bos_token_id)
        for i in range(max_new_tokens):
            logits = model(input_ids[:, -max_seq_len:])
            logits = logits[:, -1, :]
            if do_sample:
                logits = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(logits, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if (input_ids == tokenizer.eos_token_id).any(dim=-1).all():
                break

        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return texts

def generate_hf(model, tokenizer, device, max_seq_len=256, max_new_tokens=256, batch_size=1, do_sample=True):
    model.eval()
    with torch.no_grad():
        input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device).fill_(tokenizer.bos_token_id)
        for i in range(max_new_tokens):
            output = model(input_ids=input_ids[:, -max_seq_len:])
            logits = output.logits[:, -1, :]
            if do_sample:
                logits = torch.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(logits, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            if (input_ids == tokenizer.eos_token_id).any(dim=-1).all():
                break

        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        return texts


def evaluate_grid(model_type, model, ans_token_id, batch, device):
    model.eval()
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        if model_type == "hf":
            output = model(input_ids=input_ids)
            logits = output.logits
        else:  # model_type == "custom"
            logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        # compute accuracy only on ans tokens
        ans_token_ids = (input_ids == ans_token_id).nonzero(as_tuple=True)
        logs = logits[ans_token_ids]
        labs = labels[ans_token_ids]
        preds = torch.argmax(logs, dim=-1)
        acc = (preds == labs).float().mean().item()

        return loss, acc


def main(args):
    output_dir = Path(f"outputs/{datetime.strftime(datetime.now(), '%Y-%m-%d/%H-%M-%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)


    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.wandb:
        import wandb

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)

    if args.dataset == "grid":
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.dataset == "shakespeare":
        ds, vocab_size = get_shakespeare(args.max_seq_len, tokenizer=tokenizer.name_or_path)
    elif args.dataset == "wikitext":
        ds, vocab_size = get_wikitext(args.max_seq_len, tokenizer=tokenizer.name_or_path)
    elif args.dataset == "slim_pajama":
        ds, vocab_size = get_slim_pajama(args.max_seq_len, tokenizer=tokenizer.name_or_path)
    elif args.dataset == "repetition":
        ds, vocab_size = get_repetition_task(args.max_seq_len)
    elif args.dataset == "grid":
        ds, vocab = get_grid_task(args.max_seq_len, num_workers=min(32, os.cpu_count()), num_samples=100_000)
        vocab_size = len(vocab)
        ans_token_id = vocab.index("<ans>")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.attn_type == "gpt2":
        model_type = "hf"
        model_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=args.max_seq_len,
            d_embd=args.d_model,
            mlp_expansion_factor=args.mlp_expansion_factor,
            n_head=args.num_heads,
            n_layer=args.num_layers,
            attn_pdrop=args.attention_dropout,
            resid_pdrop=args.residual_dropout,
            layer_norm_eps=args.ln_eps,
            bos_token_id=tokenizer.bos_token_id if tokenizer else 0,
            eos_token_id=tokenizer.eos_token_id if tokenizer else 0,
        )
        model = GPT2LMHeadModel(model_config).to(device)
    else:
        model_type = "custom"
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

    val_ds = ds["validation"] if "validation" in ds else ds["test"]
    train_dl = DataLoader(ds["train"], batch_size=args.batch_size, pin_memory=True, num_workers=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98), eps=1e-12)
    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype != "fp32")

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project="barrel-rec",
            config=args,
            tags=[args.attn_type, args.dataset] + ([tokenizer.name_or_path] if tokenizer else []),
        )
        wandb.watch(model, log_freq=100)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    global_step = 0
    step = 0
    total_tokens = 0
    steps_per_epoch = len(train_dl) + min(args.eval_batches, len(val_dl))
    with tqdm.tqdm(total=args.epochs * steps_per_epoch) as pbar:
        stats = {}
        for epoch in range(args.epochs):
            stats["epoch"] = epoch

            for batch in train_dl:
                model.train()
                start_time = time.time()
                with torch.cuda.amp.autocast(dtype=dtype, enabled=args.dtype != "fp32"):
                    loss = get_loss(model_type, model, batch, device)
                scaler.scale(loss).backward()

                # plot gradient norm
                norms = []
                for p in model.parameters():
                    if p.grad is not None:
                        norms.append(p.grad.norm().float().item())

                if (global_step + 1) % args.accumulate_grad_batches == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    step += 1

                time_taken = time.time() - start_time
                toks_per_sec = batch["input_ids"].numel() / time_taken
                total_tokens += batch["input_ids"].numel()

                if not torch.isnan(loss).any():
                    stats["train_loss"] = 0.9 * stats.get("train_loss", loss.item()) + 0.1 * loss.item()

                    if args.wandb:
                        wandb.log({
                            "train_loss": loss.item(),
                            "train_loss_ema": stats["train_loss"],
                            "grad_norm": np.mean(norms),
                            "toks_per_sec": toks_per_sec,
                            "total_tokens": total_tokens,
                            "global_step": global_step,
                            "epoch": epoch,
                        }, step=step)

                global_step += 1
                pbar.update(1)
                pbar.set_postfix(stats)

                if (global_step + 1) % args.eval_freq == 0:
                    model.eval()
                    with torch.no_grad():
                        losses = []
                        accs = []
                        for i, batch in enumerate(val_dl):
                            if i >= args.eval_batches:
                                break
                            if args.dataset in ["grid"]:
                                loss, acc = evaluate_grid(model_type, model, ans_token_id, batch, device)
                                accs.append(acc)
                            else:
                                loss = get_loss(model_type, model, batch, device)

                            losses.append(loss.item())

                            pbar.update(1)
                        
                        if len(losses) > 0:
                            stats["val_loss"] = sum(losses) / len(losses)
                        if len(accs) > 0:
                            stats["val_acc"] = sum(accs) / len(accs)
                        pbar.set_postfix(stats)

                        if args.dataset not in ["repetition", "grid"]:
                            samples = generate(
                                model_type,
                                model,
                                tokenizer=tokenizer,
                                device=device,
                                max_seq_len=args.max_seq_len,
                                max_new_tokens=128,
                                batch_size=4,
                                do_sample=True,
                            )

                            tqdm.tqdm.write(f"------------ {global_step=} ({datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')}) ------------")
                            for i, sample in enumerate(samples):
                                    tqdm.tqdm.write(f"[{i}] {sample}")

                            if args.wandb:
                                gen_samples = wandb.Table(columns=["step", "loss", "text"])
                                for i, sample in enumerate(samples):
                                    gen_samples.add_data(global_step, stats["val_loss"], sample)
                                wandb.log({"samples": gen_samples}, step=step)



                    if args.wandb:
                        wandb.log({
                            key: stats[key] for key in ["val_loss", "val_acc"] if key in stats
                        }, step=step)
                    
                    torch.save(model.state_dict(), output_dir / "model.pt")

    torch.save(model.state_dict(), output_dir / "model.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)
