import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import ID_TO_TOKEN, detokenize, make_batch
from src.model import GPTConfig, TinyGPT, masked_cross_entropy

NUM_STEPS = 500
LOG_EVERY = 50
LR = 3e-3
BATCH_SIZE = 8
NUM_DIGITS = 2
SEED = 0


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_tokens(ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[i] for i in ids)


def main() -> None:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = pick_device()
    print(f"device: {device}")

    cfg = GPTConfig()
    model = TinyGPT(cfg).to(device)
    print(f"params: {model.num_params():,}")

    batch = make_batch(
        batch_size=BATCH_SIZE, num_digits=NUM_DIGITS, rng=rng, device=device
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0
    )

    model.train()
    final_loss = float("nan")
    for step in range(NUM_STEPS + 1):
        logits = model(batch.x)
        loss = masked_cross_entropy(logits, batch.y, batch.loss_mask)

        if step % LOG_EVERY == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

        if step == NUM_STEPS:
            final_loss = loss.item()
            break

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    model.eval()
    with torch.no_grad():
        logits = model(batch.x)
        preds = logits.argmax(dim=-1)

    x_cpu = batch.x.cpu().tolist()
    y_cpu = batch.y.cpu().tolist()
    pred_cpu = preds.cpu().tolist()
    mask_cpu = batch.loss_mask.cpu().tolist()

    correct = 0
    samples_to_show = 4
    for i in range(BATCH_SIZE):
        # rebuild original sequence (x is sequence[:-1], y is sequence[1:]) for display
        original = [x_cpu[i][0]] + y_cpu[i]
        # extract target and prediction at the supervised positions only
        target_answer = [
            y_cpu[i][t] for t in range(len(y_cpu[i])) if mask_cpu[i][t]
        ]
        predicted_answer = [
            pred_cpu[i][t] for t in range(len(pred_cpu[i])) if mask_cpu[i][t]
        ]
        is_correct = predicted_answer == target_answer
        correct += int(is_correct)
        if i < samples_to_show:
            mark = "OK" if is_correct else "X"
            print(
                f"sample: {detokenize(original):18s}"
                f"  target: {decode_tokens(target_answer):8s}"
                f"  pred: {decode_tokens(predicted_answer):8s}"
                f"  [{mark}]"
            )

    print(f"exact-match accuracy on {BATCH_SIZE} fixed examples: {correct}/{BATCH_SIZE}")
    print(f"final loss: {final_loss:.4f}")

    assert final_loss < 0.01, f"final loss {final_loss:.4f} is too high"
    assert correct == BATCH_SIZE, f"only {correct}/{BATCH_SIZE} examples correct"
    print("OK")


if __name__ == "__main__":
    main()
