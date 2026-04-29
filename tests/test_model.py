import torch

from src.data import EQ_ID, VOCAB_SIZE
from src.model import CalculatorHook, GPTConfig, HardAddSTE, TinyGPT, masked_cross_entropy


def _small_cfg() -> GPTConfig:
    return GPTConfig(n_embd=32, n_layer=2, n_head=2, block_size=8)


def _small_calculator_cfg(mode: str = "add") -> GPTConfig:
    return GPTConfig(
        n_embd=32,
        n_layer=2,
        n_head=2,
        block_size=8,
        calculator_enabled=True,
        calculator_mode=mode,
        calculator_hook_after_layer=1,
        calculator_operand_vocab_size=10,
        calculator_result_vocab_size=19,
    )


def test_forward_shape() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_cfg())
    x = torch.randint(0, VOCAB_SIZE, (2, 8))

    logits = model(x)

    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert logits.dtype == torch.float32


def test_causal_mask_does_not_leak_future_tokens() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_cfg())
    model.eval()

    a = torch.randint(0, VOCAB_SIZE, (1, 8))
    b = a.clone()
    # change only the last position; logits at earlier positions must be unchanged
    b[0, -1] = (a[0, -1] + 1) % VOCAB_SIZE

    with torch.no_grad():
        la = model(a)
        lb = model(b)

    assert torch.allclose(la[:, :-1, :], lb[:, :-1, :], atol=1e-6)


def test_calculator_off_preserves_forward_and_generate_contracts() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="off"))
    x = torch.randint(0, VOCAB_SIZE, (2, 8))

    logits = model(x)
    generated = model.generate(x[:, :3], max_new_tokens=2)

    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert generated.shape == (2, 5)


def test_hard_add_ste_forward_returns_sum_class() -> None:
    a_logits = torch.full((1, 10), -10.0)
    b_logits = torch.full((1, 10), -10.0)
    a_logits[0, 3] = 10.0
    b_logits[0, 4] = 10.0

    result = HardAddSTE.apply(a_logits, b_logits)

    assert result.shape == (1, 19)
    assert result.argmax(dim=-1).item() == 7
    assert result[0, 7].item() == 1.0


def test_hard_add_ste_backward_routes_sum_gradients_to_operand_logits() -> None:
    a_logits = torch.full((1, 10), -10.0, requires_grad=True)
    b_logits = torch.full((1, 10), -10.0, requires_grad=True)
    with torch.no_grad():
        a_logits[0, 3] = 10.0
        b_logits[0, 4] = 10.0

    result = HardAddSTE.apply(a_logits, b_logits)
    weights = torch.arange(19, dtype=result.dtype)
    (result * weights).sum().backward()

    assert torch.equal(a_logits.grad[0], torch.arange(4, 14, dtype=result.dtype))
    assert torch.equal(b_logits.grad[0], torch.arange(3, 13, dtype=result.dtype))


def test_calculator_injection_is_localized_to_equals_positions() -> None:
    torch.manual_seed(0)
    hook = CalculatorHook(_small_calculator_cfg(mode="add"))
    with torch.no_grad():
        hook.input_proj.weight.zero_()
        hook.input_proj.bias.fill_(-10.0)
        hook.input_proj.bias[3] = 10.0
        hook.input_proj.bias[10 + 4] = 10.0
        hook.output_proj.weight.fill_(1.0)

    h = torch.randn(1, 5, 32)
    tokens = torch.tensor([[1, 2, EQ_ID, 3, 4]])

    injection = hook(h, tokens)

    assert torch.all(injection[0, :2] == 0)
    assert torch.all(injection[0, 3:] == 0)
    assert torch.all(injection[0, 2] != 0)


def test_causal_mask_does_not_leak_future_tokens_with_calculator_enabled() -> None:
    torch.manual_seed(0)
    model = TinyGPT(_small_calculator_cfg(mode="add"))
    model.eval()

    a = torch.randint(0, VOCAB_SIZE, (1, 8))
    a[0, 3] = EQ_ID
    b = a.clone()
    b[0, -1] = (a[0, -1] + 1) % VOCAB_SIZE

    with torch.no_grad():
        la = model(a)
        lb = model(b)

    assert torch.allclose(la[:, :-1, :], lb[:, :-1, :], atol=1e-6)


def test_masked_cross_entropy_ignores_unmasked_positions() -> None:
    B, T, V = 1, 3, VOCAB_SIZE
    logits = torch.full((B, T, V), -10.0)
    # position 0: perfect prediction for class 1
    logits[0, 0, 1] = 10.0
    # positions 1 and 2: very wrong (would dominate loss if not masked)
    logits[0, 1, 0] = 10.0
    logits[0, 2, 0] = 10.0

    targets = torch.tensor([[1, 5, 7]])
    mask = torch.tensor([[True, False, False]])

    loss = masked_cross_entropy(logits, targets, mask)

    assert loss.item() < 1e-3
