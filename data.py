from typing import Optional

import torch
from torch.utils.data import Dataset

try:
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
except ImportError:
    GPT2TokenizerFast = None
    load_dataset = None

class ToyCharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 128, vocab: Optional[str] = None):
        super().__init__()
        if vocab is None:
            vocab = ''.join(sorted(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.ids = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.ids) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.ids[idx: idx + self.seq_len]
        y = self.ids[idx + 1: idx + 1 + self.seq_len]
        return {'input_ids': x, 'labels': y}

class HFTextDataset(Dataset):
    def __init__(self, name: str, split: str, tokenizer, seq_len: int):
        super().__init__()
        if load_dataset is None or tokenizer is None:
            raise ImportError("Please install transformers and datasets: pip install transformers datasets")
        if name == 'wikitext2':
            d = load_dataset('wikitext', 'wikitext-2-raw-v1')[split]
        elif name == 'wikitext103':
            d = load_dataset('wikitext', 'wikitext-103-raw-v1')[split]
        elif name == 'openwebtext':
            d = load_dataset('openwebtext')[split]
        else:
            raise ValueError("Unsupported dataset")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.examples = []
        # Pack text into contiguous token stream and split into windows
        ids = []
        for rec in d:
            txt = rec['text'] if 'text' in rec else ''
            if txt and not txt.isspace():
                enc = tokenizer.encode(txt)
                ids.extend(enc)
        ids = torch.tensor(ids, dtype=torch.long)
        n = (len(ids) - 1) // seq_len
        ids = ids[: n * seq_len + 1]
        for i in range(n):
            x = ids[i*seq_len:(i+1)*seq_len]
            y = ids[i*seq_len+1:(i+1)*seq_len+1]
            self.examples.append({'input_ids': x, 'labels': y})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
