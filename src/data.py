import random
from dataclasses import dataclass

import torch

VOCAB: list[str] = [str(d) for d in range(10)] + ["+", "=", "<eos>", "<pad>"]
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN: dict[int, str] = {i: tok for i, tok in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)

PLUS_ID: int = TOKEN_TO_ID["+"]
EQ_ID: int = TOKEN_TO_ID["="]
EOS_ID: int = TOKEN_TO_ID["<eos>"]
PAD_ID: int = TOKEN_TO_ID["<pad>"]


@dataclass(frozen=True)
class ArithmeticBatch:
    x: torch.Tensor
    y: torch.Tensor
    loss_mask: torch.Tensor


def tokenize(s: str) -> list[int]:
    ids: list[int] = []
    i = 0
    while i < len(s):
        if s.startswith("<eos>", i):
            ids.append(EOS_ID)
            i += len("<eos>")
        elif s.startswith("<pad>", i):
            ids.append(PAD_ID)
            i += len("<pad>")
        else:
            ids.append(TOKEN_TO_ID[s[i]])
            i += 1
    return ids


def detokenize(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def max_sequence_length(num_digits: int) -> int:
    return (num_digits * 2) + 2 + (num_digits + 1) + 1


def generate_sample(
    num_digits: int, rng: random.Random, fixed_width: bool = True
) -> list[int]:
    high = 10**num_digits - 1
    a = rng.randint(0, high)
    b = rng.randint(0, high)
    if fixed_width:
        return tokenize(f"{a:0{num_digits}d}+{b:0{num_digits}d}={a + b}<eos>")
    return tokenize(f"{a}+{b}={a + b}<eos>")


def generate_batch(
    batch_size: int, num_digits: int, rng: random.Random, fixed_width: bool = True
) -> list[list[int]]:
    return [
        generate_sample(num_digits, rng, fixed_width=fixed_width)
        for _ in range(batch_size)
    ]


def make_loss_mask(ids: list[int]) -> list[int]:
    mask = [0] * len(ids)
    try:
        eq_pos = ids.index(EQ_ID)
    except ValueError:
        return mask
    for i in range(eq_pos + 1, len(ids)):
        mask[i] = 1
    return mask


def pad_sequence(ids: list[int], length: int, pad_id: int = PAD_ID) -> list[int]:
    if len(ids) > length:
        raise ValueError(f"sequence length {len(ids)} exceeds max length {length}")
    return ids + [pad_id] * (length - len(ids))


def make_batch(
    batch_size: int,
    num_digits: int,
    rng: random.Random,
    fixed_width: bool = True,
    device: str | torch.device | None = None,
) -> ArithmeticBatch:
    samples = generate_batch(
        batch_size=batch_size,
        num_digits=num_digits,
        rng=rng,
        fixed_width=fixed_width,
    )
    seq_len = max(len(ids) for ids in samples)
    if fixed_width:
        seq_len = max_sequence_length(num_digits)

    padded = [pad_sequence(ids, seq_len) for ids in samples]
    masks = [pad_sequence(make_loss_mask(ids), seq_len, pad_id=0) for ids in samples]

    tokens = torch.tensor(padded, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)

    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )
