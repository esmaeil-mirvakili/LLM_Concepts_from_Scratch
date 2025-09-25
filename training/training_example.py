import math
from dataclasses import dataclass
from typing import Tuple, List

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# import your Trainer
from training.training_utils import Trainer


# ----------------------- Toy tokenizer & dataset -----------------------


class CharTokenizer:
    def __init__(self, text: str):
        vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CharSequenceDataset(Dataset):
    def __init__(self, data: Tensor, block_size: int):
        """
        data: 1D LongTensor of token ids
        returns (x, y) where y is x shifted by 1 (next-token prediction)
        """
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, self.data.size(0) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


# ----------------------- Tiny GPT-ish model -----------------------


@dataclass
class GPTConfig:
    vocab_size: int
    n_embd: int = 64
    n_head: int = 2
    n_layer: int = 2
    block_size: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        # causal mask (buffer so it moves with module)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = (
            self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        )  # B,h,T,hd
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # B,h,T,T
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        y = att @ v  # B,h,T,hd
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

    def forward(self, idx: Tensor) -> Tensor:
        # idx: (B, T) int64
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(
            0
        )  # 1,T
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # B,T,V
        return logits

    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int = 100) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # last time step
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # B,1
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ----------------------- Loss & metric -----------------------


def sequence_ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    logits: (B, T, V), targets: (B, T)
    """
    B, T, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
        reduction="mean",
    )
    return loss


def token_accuracy(logits: Tensor, targets: Tensor) -> float:
    """
    Simple next-token accuracy: argmax(logits) vs targets at each position
    """
    if not (isinstance(logits, torch.Tensor) and isinstance(targets, torch.Tensor)):
        return 0.0
    pred = logits.argmax(dim=-1)  # B,T
    correct = (pred == targets).float().mean().item()
    return correct


# ----------------------- Collate / prepare_batch -----------------------


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def prepare_batch_fn(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


# ----------------------- Main script -----------------------


def main():
    # Use a very small dataset from Hugging Face datasets
    dataset = load_dataset("tiny_shakespeare", split="train", trust_remote_code=True)
    texts = dataset["text"]
    # Keep only a handful of lines to stay lightweight on CPU
    texts = texts[:50] if isinstance(texts, list) else [str(texts)]
    text_data = "\n".join(texts)

    tokenizer = CharTokenizer(text_data)
    ids = torch.tensor(tokenizer.encode(text_data), dtype=torch.long)

    block_size = 64
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=64,
        n_head=2,
        n_layer=2,
        block_size=block_size,
        dropout=0.0,
    )
    model = TinyGPT(cfg)

    # Dataset & loader
    dataset = CharSequenceDataset(ids, block_size=block_size)
    loader = DataLoader(
        dataset,
        batch_size=32,  # small enough for CPU
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01
    )

    # Trainer
    trainer = Trainer(
        model=model,
        device="cpu",
        optimizer=optimizer,
        loss_fn=sequence_ce_loss,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        lr_scheduler=None,  # keep it simple; or use torch.optim.lr_scheduler.StepLR(...)
        scheduler_step_policy="epoch",
        prepare_batch_fn=prepare_batch_fn,
        callbacks=[],  # plug your own loggers if you want
        metric_fns={"acc": token_accuracy},  # logs epoch-level accuracy
        evaluate_every=0,  # skip eval during training (no val set)
        evaluate_max_steps=None,
        log_every=5,
        use_amp=False,  # CPU: keep AMP off
        amp_dtype=None,
    )

    # Train
    history = trainer.train(train_dataloader=loader, val_dataloader=None, epochs=20)
    print("History:", history[-1])

    # Sample a bit of text
    model.eval()
    start = "to be "
    ctx = torch.tensor([tokenizer.encode(start)], dtype=torch.long)
    out = model.generate(ctx, max_new_tokens=120)
    print("\n=== SAMPLE ===")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
