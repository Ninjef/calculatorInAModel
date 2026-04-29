import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.data import EOS_ID, detokenize, tokenize
from src.model import GPTConfig, TinyGPT


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(run_dir: Path, device: str) -> TinyGPT:
    config = json.loads((run_dir / "config.json").read_text())
    cfg = GPTConfig(**config["model"])
    model = TinyGPT(cfg).to(device)

    checkpoint = torch.load(
        run_dir / "final_weights.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def complete(model: TinyGPT, prompt: str, max_new_tokens: int, device: str) -> str:
    ids = tokenize(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(x, max_new_tokens=max_new_tokens)[0].tolist()

    new_ids = generated[len(ids) :]
    if EOS_ID in new_ids:
        new_ids = new_ids[: new_ids.index(EOS_ID) + 1]
    return prompt + detokenize(new_ids), detokenize(new_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try a trained addition model.")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing config.json and final_weights.pt.",
    )
    parser.add_argument(
        "problems",
        nargs="*",
        help='Prompts like "07+05=". If omitted, starts an interactive prompt.',
    )
    parser.add_argument("--max-new-tokens", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device()
    model = load_model(args.run_dir, device)
    print(f"loaded {args.run_dir} on {device}")

    problems = args.problems
    if problems:
        for problem in problems:
            full, answer = complete(model, problem, args.max_new_tokens, device)
            print(f"{problem} -> {answer}  ({full})")
        return

    print('Enter prompts like "07+05=". Press Ctrl-D to quit.')
    while True:
        try:
            problem = input("> ").strip()
        except EOFError:
            print()
            break
        if not problem:
            continue
        full, answer = complete(model, problem, args.max_new_tokens, device)
        print(f"{answer}  ({full})")


if __name__ == "__main__":
    main()
