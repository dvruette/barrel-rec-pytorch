import os
from pathlib import Path
import pickle

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from grid import GridFactory


def _get_encode_and_chunk_fn(tokenizer_id: str = "gpt2", seq_len: int = 256):
    def encode_and_chunk(batch):
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
    return encode_and_chunk


def process_dataset(ds, seq_len: int = 256, tokenizer: str = "gpt2"):
    tokenizer_id = tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    ds = ds.map(
        _get_encode_and_chunk_fn(tokenizer_id, seq_len),
        batched=True,
        remove_columns=["text"],
        num_proc=min(32, os.cpu_count()),
        # cache_file_names={split: f"wikitext-103-raw-v1_{split=}_{tokenizer=}_{seq_len=}".replace("/", "-") for split in ds},
    )
    return ds, tokenizer.vocab_size


def get_shakespeare(seq_len: int = 256, tokenizer: str = "gpt2"):
    ds = load_dataset("tiny_shakespeare")
    ds, vocab_size = process_dataset(ds, seq_len, tokenizer)
    return ds.with_format("torch"), vocab_size


def get_wikitext(seq_len: int = 256, tokenizer: str = "gpt2"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    ds, vocab_size = process_dataset(ds, seq_len, tokenizer)
    return ds.with_format("torch"), vocab_size


def get_slim_pajama(seq_len: int = 256, tokenizer: str = "gpt2"):
    ds = load_dataset("DKYoon/SlimPajama-6B")
    ds = ds.remove_columns(["meta", "__index_level_0__"])
    ds, vocab_size = process_dataset(ds, seq_len, tokenizer)
    return ds.with_format("torch"), vocab_size


def get_repetition_task(seq_len: int = 256, vocab_size: int = 1000, num_samples: int = 100_000, test_size: float = 0.2):
    tokens = torch.randint(0, vocab_size, (num_samples, seq_len // 2))
    tokens = torch.cat([tokens, tokens], dim=-1)
    input_ids = tokens[:, :-1]
    labels = tokens[:, 1:]

    ds = Dataset.from_dict({"input_ids": input_ids.tolist(), "labels": labels.tolist()})
    ds = ds.train_test_split(test_size=test_size)
    return ds.with_format("torch"), vocab_size


def get_grid_task(max_seq_len: int = 256, grid_size: int = 6, num_samples: int = 250_000, test_size: float = 0.1, num_workers: int = 1, use_cache: bool = True):
    cache_file = Path(f"/tmp/barrel-rec/grid_task_{grid_size=}_{num_samples=}_{max_seq_len=}.pkl")
    if use_cache and cache_file.exists():
        with cache_file.open("rb") as f:
            input_ids, labels, vocab = pickle.load(f)
    else:
        grid_factory = GridFactory(size=grid_size)
        samples = grid_factory.generate_samples(num_samples, show_progress=True, num_workers=num_workers)
        words = [x.strip().split() for x in samples]
        max_len = max(len(xs) for xs in words)
        if max_len > max_seq_len:
            raise ValueError(f"max_seq_len={max_seq_len} is too small for the grid task. The longest sequence is {max_len} words long.")
        
        vocab = ["<pad>"] + sorted(set([x for xs in words for x in xs]))
        word_to_id = {w: i for i, w in enumerate(vocab)}
        tokens = [[word_to_id[w]  for w in xs] + [word_to_id["<pad>"]] * (max_len - len(xs) + 1) for xs in words]
        input_ids = [xs[:-1] for xs in tokens]
        labels = [xs[1:] for xs in tokens]

        if use_cache:
            cache_file.parent.mkdir(exist_ok=True, parents=True)
            with cache_file.open("wb") as f:
                pickle.dump((input_ids, labels, vocab), f)

    ds = Dataset.from_dict({"input_ids": input_ids, "labels": labels})
    ds = ds.train_test_split(test_size=test_size)
    return ds.with_format("torch"), vocab
