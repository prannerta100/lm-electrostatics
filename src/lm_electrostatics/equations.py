import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    """Return 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name="gpt2", device=None, dtype=None):
    """
    Load a pre-trained causal LM in eval mode.
    Returns (model, tokenizer).

    Args:
        model_name: HuggingFace model name (e.g. "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        device: Device string. Auto-detected if None.
        dtype: torch dtype (e.g. torch.float32, torch.bfloat16). None = model default.
    """
    if device is None:
        device = get_device()

    load_kwargs = {"attn_implementation": "eager"}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _get_transformer_backbone(model):
    """Get the transformer backbone from a causal LM model (handles different architectures)."""
    # GPT-2 style
    if hasattr(model, "transformer"):
        return model.transformer
    # LLaMA / Mistral / Qwen style
    if hasattr(model, "model"):
        return model.model
    raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


def _get_embed_dim(model):
    """Get hidden dimension from model config."""
    config = model.config
    for attr in ("n_embd", "hidden_size", "d_model"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot determine hidden size from config: {type(config).__name__}")


def _get_layers(model):
    """Get the list of transformer layers/blocks."""
    backbone = _get_transformer_backbone(model)
    # GPT-2: backbone.h
    if hasattr(backbone, "h"):
        return backbone.h
    # LLaMA / Mistral / Qwen: backbone.layers
    if hasattr(backbone, "layers"):
        return backbone.layers
    raise ValueError(f"Cannot find transformer layers in {type(backbone).__name__}")


def _get_num_layers(model):
    """Get number of transformer layers."""
    return len(_get_layers(model))


def get_embedding(model, input_ids):
    """
    Compute X_0 = embedding(input_ids).

    Handles GPT-2 (wte + wpe) and LLaMA-style (embed_tokens) architectures.

    Args:
        model: CausalLM model
        input_ids: (1, S) tensor

    Returns:
        X_0 as (S, H) tensor, detached, with requires_grad=True, in float32
    """
    backbone = _get_transformer_backbone(model)
    S = input_ids.shape[1]
    device = input_ids.device

    if hasattr(backbone, "wte") and hasattr(backbone, "wpe"):
        # GPT-2 style: token embedding + position embedding
        positions = torch.arange(S, device=device)
        x0 = backbone.wte(input_ids) + backbone.wpe(positions)
    elif hasattr(backbone, "embed_tokens"):
        # LLaMA / Mistral / Qwen style
        x0 = backbone.embed_tokens(input_ids)
    else:
        raise ValueError(f"Cannot find embedding layers in {type(backbone).__name__}")

    x0 = x0.squeeze(0).detach().clone()
    x0.requires_grad_(True)
    return x0


def get_layer_output_fn(model, layer_idx):
    """
    Return a function f(x0_flat) -> x_l_flat that maps flattened X_0 to flattened X_l.

    This is differentiable w.r.t. x0_flat. Does not use hooks.
    Manually iterates through transformer blocks 0..layer_idx.

    Args:
        model: CausalLM model
        layer_idx: int, 0-indexed (0 to L-1)

    Returns:
        Callable that takes (S*H,) tensor and returns (S*H,) tensor
    """
    H = _get_embed_dim(model)
    layers = _get_layers(model)

    def fn(x0_flat):
        S = x0_flat.shape[0] // H
        hidden = x0_flat.view(1, S, H)

        for i in range(layer_idx + 1):
            block = layers[i]
            outputs = block(hidden)
            if isinstance(outputs, torch.Tensor):
                hidden = outputs
            else:
                hidden = outputs[0]

        return hidden.squeeze(0).reshape(-1)

    return fn


def compute_perplexity(model, tokenizer, text):
    """
    Compute perplexity of text under the model.
    PPL = exp(cross-entropy loss).
    """
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()
