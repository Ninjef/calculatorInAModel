import random

import pytest
import torch

from src.data import (
    EOS_ID,
    PAD_ID,
    detokenize,
    make_batch,
    make_loss_mask,
    max_sequence_length,
    pad_sequence,
    tokenize,
)


def test_tokenize_round_trip_special_tokens() -> None:
    ids = tokenize("07+05=12<eos><pad>")

    assert detokenize(ids) == "07+05=12<eos><pad>"
    assert ids[-2:] == [EOS_ID, PAD_ID]


def test_loss_mask_only_scores_answer_and_eos() -> None:
    ids = tokenize("07+05=12<eos>")

    assert make_loss_mask(ids) == [0, 0, 0, 0, 0, 0, 1, 1, 1]


def test_pad_sequence_rejects_too_long_input() -> None:
    with pytest.raises(ValueError):
        pad_sequence([1, 2, 3], length=2)


def test_make_fixed_width_batch_shapes_and_padding() -> None:
    batch = make_batch(batch_size=16, num_digits=2, rng=random.Random(0))
    expected_time = max_sequence_length(2) - 1

    assert batch.x.shape == (16, expected_time)
    assert batch.y.shape == (16, expected_time)
    assert batch.loss_mask.shape == (16, expected_time)
    assert batch.x.dtype == torch.long
    assert batch.y.dtype == torch.long
    assert batch.loss_mask.dtype == torch.bool
    assert torch.all(batch.loss_mask[:, :5] == 0)
    assert torch.all(batch.loss_mask[batch.y == PAD_ID] == 0)


def test_make_natural_width_batch_pads_to_longest_sample() -> None:
    batch = make_batch(
        batch_size=32,
        num_digits=2,
        rng=random.Random(1),
        fixed_width=False,
    )

    assert batch.x.shape[0] == 32
    assert batch.x.shape == batch.y.shape == batch.loss_mask.shape
    assert torch.any(batch.x == PAD_ID)
