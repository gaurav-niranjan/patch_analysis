import math
import torch
from PIL import Image
from transformers import AutoProcessor
from typing import Dict, Tuple

def make_capture_hook(storage, layer_idx, image_positions):
    def hook(module, args):
        hidden_states = args[0]
        seq_len = hidden_states.shape[1]
        
        # Only capture during prefill (full sequence), skip decode steps
        if seq_len > 1:
            storage[layer_idx] = hidden_states[0, image_positions, :].detach().clone()
        
        return None
    
    return hook

def clean_capture(model, inputs, image_positions, num_deepstack=3, generate=False, max_new_tokens=128):
    """
    Run a clean forward pass and capture hidden states at all injection points.
    Returns (captured, outputs):
      - captured: {layer_idx: tensor of shape (num_image_tokens, hidden_dim)}
      - outputs: model outputs or generated token ids
    """
    text_model = model.model.language_model
    hook_layers = list(range(num_deepstack + 1))
    
    captured = {}
    handles = []
    
    for layer_idx in hook_layers:
        hook = make_capture_hook(captured, layer_idx, image_positions)
        handle = text_model.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=False)
        handles.append(handle)
    
    try:
        with torch.no_grad():
            if generate:
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                outputs = model(**inputs)
    finally:
        for h in handles:
            h.remove()
    
    return captured, outputs


def compute_replacements(captured, strategy="inpainted"):

    """
    Given captured clean hidden states, compute replacement values.
    
    captured: {layer_idx: (num_image_tokens, hidden_dim)}
    Returns: {layer_idx: (hidden_dim,)} or whatever shape your strategy produces
    """
    replacements = {}
    
    for layer_idx, features in captured.items():
        if strategy == "mean":
            replacements[layer_idx] = features.mean(dim=0)        # (hidden_dim,)
        elif strategy == "zeros":
            replacements[layer_idx] = torch.zeros_like(features[0])
        elif strategy == "random_token":
            rand_idx = torch.randint(0, features.shape[0], (1,)).item()
            replacements[layer_idx] = features[rand_idx]
        elif strategy == "inpainted":
            replacements[layer_idx] = features #is of shape (num_image_tokens, hidden_dim)
    return replacements

#Ablation hook:
def make_ablation_hook(selected_positions, replacement_value, per_position=False, all_image_positions=None):
    if per_position:
        # Build mapping: sequence_pos -> index in replacement tensor
        if all_image_positions is None:
            raise ValueError("all_image_positions must be provided when per_position=True")
        pos_to_idx = {pos: idx for idx, pos in enumerate(all_image_positions)}
        replacement_indices = [pos_to_idx[p] for p in selected_positions]

    def hook(module, args):
        hidden_states = args[0]
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states.clone()
            if per_position:
                hidden_states[0, selected_positions, :] = replacement_value[replacement_indices, :]
            else:
                hidden_states[0, selected_positions, :] = replacement_value
            return (hidden_states,) + args[1:]
        return None

    return hook

#Ablation forward pass:

def ablated_forward(
    model,
    inputs,
    selected_positions,   # which image tokens to replace
    replacements,         # {layer_idx: replacement_value}
    per_position=False,   # whether the replacement value is per-position or a single mean vector
    all_image_positions=None,  # needed if per_position is True
    generate=False,
    max_new_tokens=128,
):
    text_model = model.model.language_model
    
    handles = []
    for layer_idx, replacement in replacements.items():
        hook = make_ablation_hook(selected_positions, replacement, per_position=per_position, all_image_positions=all_image_positions)
        handle = text_model.layers[layer_idx].register_forward_pre_hook(hook, with_kwargs=False)
        handles.append(handle)
    
    try:
        with torch.no_grad():
            if generate:
                return model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                return model(**inputs)
    finally:
        for h in handles:
            h.remove()