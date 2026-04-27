import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import detokenize, generate_sample, make_batch, make_loss_mask

rng = random.Random(0)
plan = [(1, 7), (2, 7), (3, 6)]  # 20 samples total
for num_digits, count in plan:
    print(f"\n--- {num_digits}-digit samples ---")
    for _ in range(count):
        ids = generate_sample(num_digits=num_digits, rng=rng)
        mask = make_loss_mask(ids)
        print(f"text={detokenize(ids)!r:30s} ids={ids} mask={mask}")

batch = make_batch(batch_size=4, num_digits=2, rng=random.Random(1))
print("\n--- training batch ---")
print("x shape:", tuple(batch.x.shape))
print("y shape:", tuple(batch.y.shape))
print("loss_mask shape:", tuple(batch.loss_mask.shape))
