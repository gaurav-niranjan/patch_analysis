import torch
from transformers import AutoProcessor
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def downsample_image(img: Image.Image, short_side=1536):
    W, H = img.size

    # already small enough
    if min(W, H) <= short_side:
        return img

    scale = short_side / min(W, H)
    new_W = int(W * scale)
    new_H = int(H * scale)

    return img.resize((new_W, new_H), Image.Resampling.LANCZOS)

def build_prompt(example):
    return (
        #"You are answering a multiple-choice question about the image.\n\n"  
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n\n"
        f"A) {example['ans0']}\n"
        f"B) {example['ans1']}\n"
        f"C) {example['ans2']}\n\n"

        "Answer with only A, B, or C.\n"
        "Answer:"
    )

def get_textonly_input(processor, ds, id_to_index, sample_id):

    prompt = build_prompt(ds[id_to_index[sample_id]])

    conversation = [
        {
            "role": "user",
            "content": [
                    {"type": "text", "text": prompt}
                ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    return inputs


def get_input(processor, ds, id_to_index, sample_id, IMAGE_DIR, inpaited=False, style=None):

    if inpaited:
        if style == 'female_bg':
            image_path = IMAGE_DIR / sample_id / f'{style}.png'
        elif style == 'male_bg':
            image_path = IMAGE_DIR / sample_id / f'{style}.png'
        else:
            raise ValueError(f"Unknown style: {style}")
    else:
        image_path = IMAGE_DIR / sample_id / 'original.png'
    
    image = downsample_image(Image.open(image_path).convert("RGB"))
            
    prompt = build_prompt(ds[id_to_index[sample_id]])

    conversation = [
        {
            "role": "user",
            "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
        }
    ]

    #print(conversation)
    
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )

    return inputs, image_path

def find_image_positions(input_ids, IMAGE_TOKEN_ID):

        # Find all <|image_pad|> positions
    image_pad_mask = (input_ids == IMAGE_TOKEN_ID)
    image_pad_positions = torch.where(image_pad_mask)[0]

    return image_pad_positions

def build_qwen3vl_patch_to_pixel_map(
    input_ids: torch.Tensor,
    image_token_id: int,
    image_grid_thw: torch.Tensor,
    processor: AutoProcessor,
    original_height: int = None, # original image height
    original_width: int = None,  # original image width
):
    """
    Map each <|image_pad|> token back to a pixel bounding box.
    
    After merge, each token covers (patch_size * merge_size) pixels
    = 14 * 2 = 28 pixels in each spatial dimension.
    
    image_grid_thw gives (T, H, W) AFTER the merge, so:
      - H_pixels = H * patch_size * merge_size
      - W_pixels = W * patch_size * merge_size
    """

    patch_size = processor.image_processor.patch_size   # 14
    merge_size = processor.image_processor.merge_size   # 2
    effective_patch = patch_size * merge_size  # 28

    pad_positions = torch.where(input_ids == image_token_id)[0]

    mapping = {}
    token_offset = 0

    for img_idx in range(image_grid_thw.shape[0]):
        t, h_pre, w_pre = image_grid_thw[img_idx].tolist()

        # Post-merge grid dimensions
        h = h_pre // merge_size  # 30 // 2 = 15
        w = w_pre // merge_size  # 40 // 2 = 20
        num_tokens = t * h * w   # 1 * 15 * 20 = 300

        # Resized image dimensions (what the processor actually fed to the ViT)
        resized_h = h_pre * patch_size  # 30 * 14 = 420
        resized_w = w_pre * patch_size  # 40 * 14 = 560

        for local_idx in range(num_tokens):
            global_pos = pad_positions[token_offset + local_idx].item()

            spatial_idx = local_idx % (h * w)
            row = spatial_idx // w
            col = spatial_idx % w

            # Bbox in resized image coords
            x1 = col * effective_patch
            y1 = row * effective_patch
            x2 = x1 + effective_patch
            y2 = y1 + effective_patch

            info = {
                "image_idx": img_idx,
                "row": row,
                "col": col,
                "bbox_resized": (x1, y1, x2, y2),
            }

            # Scale back to original image coords
            if original_height and original_width:
                sx = original_width / resized_w
                sy = original_height / resized_h
                info["bbox_original"] = (x1*sx, y1*sy, x2*sx, y2*sy)

            mapping[global_pos] = info

        token_offset += num_tokens

    return mapping

def visualize_qwen3vl_patches(
    image,
    patch_map,
    highlight_tokens=None,  # list of LLM token indices to highlight
    use_original_coords=True,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    
    bbox_key = "bbox_original" if use_original_coords else "bbox_resized"
    
    tokens_to_show = highlight_tokens or list(patch_map.keys())
    
    for tok_pos in tokens_to_show:
        if tok_pos not in patch_map:
            continue
        info = patch_map[tok_pos]
        bbox = info.get(bbox_key, info["bbox_resized"])
        x1, y1, x2, y2 = bbox
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor='red', facecolor='red', alpha=0.25
        )
        ax.add_patch(rect)
    
    ax.set_title(f"Highlighting {len(tokens_to_show)} patches")
    plt.tight_layout()
    plt.show()

def sliding_window_on_grid(
    patch_map,
    image_grid_thw,
    k: int = None,            # fixed window size in grid cells
    frac: float = None,       # fractional window size (e.g. 0.33)
    stride: int = None,       # explicit stride; if None, auto = max(1, win_size // 2)
    merge_size: int = 2,
):
    """
    Slide a square window over the token grid.
    
    Window size is set by exactly one of:
      - k: fixed side length in grid cells
      - frac: fraction of the shorter grid axis (e.g. 0.33 → 33%)
    
    Yields (row, col, token_positions) per window.
    """
    if (k is None) == (frac is None):
        raise ValueError("Provide exactly one of k or frac")

    t, h_pre, w_pre = image_grid_thw[0].tolist()
    grid_h = h_pre // merge_size
    grid_w = w_pre // merge_size

    win = k if k is not None else max(2, round(min(grid_h, grid_w) * frac))
    step = stride if stride is not None else max(1, win // 2)

    # Build lookup: (row, col) -> LLM token index
    rc_to_token = {(info["row"], info["col"]): pos
                   for pos, info in patch_map.items()}

    for row in range(0, grid_h - win + 1, step):
        for col in range(0, grid_w - win + 1, step):
            tokens = [
                rc_to_token[(r, c)]
                for r in range(row, row + win)
                for c in range(col, col + win)
            ]
            yield row, col, tokens

