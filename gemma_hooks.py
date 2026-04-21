import torch
from PIL import Image
from transformers import AutoProcessor
from typing import Dict, Tuple


# --- Capture hooks ---

def make_capture_hook(storage, layer_idx, image_positions):
    def hook(module, args):
        hidden_states = args[0]
        if hidden_states.shape[1] > 1:  # prefill only
            storage[layer_idx] = hidden_states[0, image_positions, :].detach().clone()
        return None
    return hook


def clean_capture(model, inputs, image_positions, hook_layers=None, generate=False, max_new_tokens=128):
    """
    Run a clean forward pass and capture hidden states at specified layers.
    
    No deepstack in Gemma 3, so hook_layers is just whatever decoder layers
    you want to probe (e.g. [0, 5, 11, 23, 47] for sampling across depth).
    """
    # Gemma 3 structure: model.language_model.layers (not model.model.language_model)
    text_model = model.language_model

    if hook_layers is None:
        hook_layers = [0]  # default: just before the first decoder layer

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


# --- Replacement strategies ---

def compute_replacements(captured, strategy="inpainted"):
    """
    Given captured clean hidden states, compute replacement values.
    Same as Qwen version — strategy logic is model-agnostic.
    """
    replacements = {}
    for layer_idx, features in captured.items():
        if strategy == "mean":
            replacements[layer_idx] = features.mean(dim=0)
        elif strategy == "zeros":
            replacements[layer_idx] = torch.zeros_like(features[0])
        elif strategy == "random_token":
            rand_idx = torch.randint(0, features.shape[0], (1,)).item()
            replacements[layer_idx] = features[rand_idx]
        elif strategy == "inpainted":
            replacements[layer_idx] = features
    return replacements


# --- Ablation hooks ---

def make_ablation_hook(selected_positions, replacement_value, per_position=False, all_image_positions=None):
    if per_position:
        if all_image_positions is None:
            raise ValueError("all_image_positions must be provided when per_position=True")
        pos_to_idx = {pos: idx for idx, pos in enumerate(all_image_positions)}
        replacement_indices = [pos_to_idx[p] for p in selected_positions]

    def hook(module, args):
        hidden_states = args[0]
        if hidden_states.shape[1] > 1:
            hidden_states = hidden_states.clone()
            if per_position:
                hidden_states[0, selected_positions, :] = replacement_value[replacement_indices, :].to(hidden_states.device, hidden_states.dtype)
            else:
                hidden_states[0, selected_positions, :] = replacement_value.to(hidden_states.device, hidden_states.dtype)
            return (hidden_states,) + args[1:]
        return None

    return hook


def ablated_forward(
    model,
    inputs,
    selected_positions,
    replacements,
    per_position=False,
    all_image_positions=None,
    generate=False,
    max_new_tokens=128,
):
    """
    Single ablation point — replace image token hidden states before specified layers.
    
    For Gemma 3, replacements typically only has layer 0 (the sole injection point),
    """
    
    text_model = model.language_model

    handles = []
    for layer_idx, replacement in replacements.items():
        hook = make_ablation_hook(
            selected_positions, replacement,
            per_position=per_position,
            all_image_positions=all_image_positions,
        )
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

def get_image_embeds(model, inputs, image_token_id=262144):
    """
    Capture all image embeddings as seen by layer 0 of the decoder.
    Returns: (num_image_tokens, hidden_dim) — e.g. (256, 3840)
    """
    input_ids = inputs["input_ids"][0]
    all_image_positions = torch.where(input_ids == image_token_id)[0].tolist()

    captured, _ = clean_capture(model, inputs, all_image_positions, hook_layers=[0])
    return captured[0], all_image_positions