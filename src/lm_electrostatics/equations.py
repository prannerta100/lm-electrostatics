import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(device=None):
    """
    Load pre-trained 'gpt2' (124M) in eval mode.
    Returns (model, tokenizer).
    Moves model to device. Freezes all parameters.
    """
    if device is None:
        device = get_device()
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def get_embedding(model, input_ids):
    """
    Compute X_0 = W_e[input_ids] + W_p[positions].

    Args:
        model: GPT2LMHeadModel
        input_ids: (1, S) tensor

    Returns:
        X_0 as (S, H) tensor, detached, with requires_grad=True
    """
    S = input_ids.shape[1]
    device = input_ids.device
    positions = torch.arange(S, device=device)
    x0 = model.transformer.wte(input_ids) + model.transformer.wpe(positions)
    x0 = x0.squeeze(0).detach().clone()
    x0.requires_grad_(True)
    return x0


def get_layer_output_fn(model, layer_idx):
    """
    Return a function f(x0_flat) -> x_l_flat that maps flattened X_0 to flattened X_l.

    This is differentiable w.r.t. x0_flat. Does not use hooks.
    Manually iterates through transformer blocks 0..layer_idx.

    Args:
        model: GPT2LMHeadModel
        layer_idx: int, 0-indexed (0 to L-1)

    Returns:
        Callable that takes (S*H,) tensor and returns (S*H,) tensor
    """

    def fn(x0_flat):
        S = x0_flat.shape[0] // model.config.n_embd
        H = model.config.n_embd
        hidden = x0_flat.view(1, S, H)

        for i in range(layer_idx + 1):
            block = model.transformer.h[i]
            outputs = block(hidden)
            # transformers v5+ returns a tensor directly;
            # older versions return a tuple where [0] is the hidden state
            if isinstance(outputs, torch.Tensor):
                hidden = outputs
            else:
                hidden = outputs[0]

        return hidden.squeeze(0).reshape(-1)

    return fn
