import torch
from lm_electrostatics.equations import load_model, get_embedding, get_layer_output_fn


def test_layer_output_fn_shape():
    model, tokenizer = load_model()
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    x0 = get_embedding(model, input_ids)
    x0_flat = x0.reshape(-1).detach().clone().requires_grad_(True)
    fn = get_layer_output_fn(model, layer_idx=0)
    x_l = fn(x0_flat)
    assert x_l.shape == x0_flat.shape


def test_layer_output_fn_differentiable():
    model, tokenizer = load_model()
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    x0 = get_embedding(model, input_ids)
    x0_flat = x0.reshape(-1).detach().clone().requires_grad_(True)
    fn = get_layer_output_fn(model, layer_idx=0)
    x_l = fn(x0_flat)
    loss = x_l.sum()
    loss.backward()
    assert x0_flat.grad is not None
    assert x0_flat.grad.shape == x0_flat.shape


def test_different_layers_different_outputs():
    model, tokenizer = load_model()
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    x0 = get_embedding(model, input_ids)
    x0_flat = x0.reshape(-1).detach().clone().requires_grad_(True)
    fn0 = get_layer_output_fn(model, layer_idx=0)
    fn5 = get_layer_output_fn(model, layer_idx=5)
    with torch.no_grad():
        out0 = fn0(x0_flat)
        out5 = fn5(x0_flat)
    assert not torch.allclose(out0, out5)
