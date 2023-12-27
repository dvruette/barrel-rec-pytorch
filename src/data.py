import os

from datasets import load_dataset
from transformers import AutoTokenizer


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
    ds = ds.with_format("torch")
    return ds, tokenizer.vocab_size


def get_shakespeare(seq_len: int = 256, tokenizer: str = "gpt2"):
    ds = load_dataset("tiny_shakespeare")
    return process_dataset(ds, seq_len, tokenizer)


def get_wikitext(seq_len: int = 256, tokenizer: str = "gpt2"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    return process_dataset(ds, seq_len, tokenizer)
