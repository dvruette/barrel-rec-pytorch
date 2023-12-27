import os

from datasets import load_dataset
from transformers import AutoTokenizer


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
