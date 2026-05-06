import random
from dataclasses import dataclass
from typing import Literal

import torch

VOCAB: list[str] = [str(d) for d in range(10)] + ["+", "=", "<eos>", "<pad>"]
TOKEN_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN: dict[int, str] = {i: tok for i, tok in enumerate(VOCAB)}
VOCAB_SIZE: int = len(VOCAB)

PLUS_ID: int = TOKEN_TO_ID["+"]
EQ_ID: int = TOKEN_TO_ID["="]
EOS_ID: int = TOKEN_TO_ID["<eos>"]
PAD_ID: int = TOKEN_TO_ID["<pad>"]
AnswerFormat = Literal["sum", "sum_left_operand"]
ANSWER_FORMATS: tuple[str, ...] = ("sum", "sum_left_operand")


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


def answer_target(
    a: int,
    b: int,
    num_digits: int,
    *,
    answer_format: AnswerFormat = "sum",
    fixed_width: bool = True,
) -> str:
    if answer_format == "sum":
        return f"{a + b}<eos>"
    if answer_format == "sum_left_operand":
        if fixed_width:
            return f"{a + b:0{num_digits + 1}d}{a:0{num_digits}d}<eos>"
        return f"{a + b}{a}<eos>"
    raise ValueError(f"unknown answer format: {answer_format!r}")


def max_answer_tokens(
    num_digits: int, answer_format: AnswerFormat = "sum"
) -> int:
    if answer_format == "sum":
        return (num_digits + 1) + 1
    if answer_format == "sum_left_operand":
        return (num_digits + 1) + num_digits + 1
    raise ValueError(f"unknown answer format: {answer_format!r}")


def max_sequence_length(
    num_digits: int, answer_format: AnswerFormat = "sum"
) -> int:
    return (num_digits * 2) + 2 + max_answer_tokens(num_digits, answer_format)


def generate_sample(
    num_digits: int,
    rng: random.Random,
    fixed_width: bool = True,
    answer_format: AnswerFormat = "sum",
) -> list[int]:
    high = 10**num_digits - 1
    a = rng.randint(0, high)
    b = rng.randint(0, high)
    if fixed_width:
        prompt = f"{a:0{num_digits}d}+{b:0{num_digits}d}="
    else:
        prompt = f"{a}+{b}="
    return tokenize(
        prompt
        + answer_target(
            a,
            b,
            num_digits,
            answer_format=answer_format,
            fixed_width=fixed_width,
        )
    )


def generate_batch(
    batch_size: int,
    num_digits: int,
    rng: random.Random,
    fixed_width: bool = True,
    answer_format: AnswerFormat = "sum",
) -> list[list[int]]:
    return [
        generate_sample(
            num_digits,
            rng,
            fixed_width=fixed_width,
            answer_format=answer_format,
        )
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
    answer_format: AnswerFormat = "sum",
    device: str | torch.device | None = None,
) -> ArithmeticBatch:
    samples = generate_batch(
        batch_size=batch_size,
        num_digits=num_digits,
        rng=rng,
        fixed_width=fixed_width,
        answer_format=answer_format,
    )
    seq_len = max(len(ids) for ids in samples)
    if fixed_width:
        seq_len = max_sequence_length(num_digits, answer_format=answer_format)

    padded = [pad_sequence(ids, seq_len) for ids in samples]
    masks = [pad_sequence(make_loss_mask(ids), seq_len, pad_id=0) for ids in samples]

    tokens = torch.tensor(padded, dtype=torch.long, device=device)
    loss_mask = torch.tensor(masks, dtype=torch.bool, device=device)

    return ArithmeticBatch(
        x=tokens[:, :-1],
        y=tokens[:, 1:],
        loss_mask=loss_mask[:, 1:],
    )
