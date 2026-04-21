import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from pathlib import Path
from datasets import load_dataset
import json
from PIL import Image
from patch_analysis.utils import *
from patch_analysis.prob_utils import choice_probs_ABC, probs_tensor_to_dicts
from patch_analysis.gemma_hooks import *
import pandas as pd

STYLE = "style0"
POLARITY = "polarity0"
IMAGE_DIR = Path(f"/weka/eickhoff/esx139/flux_inpainting/flux_klein/consistent_set/gemma3_12b/{STYLE}/cat_2/{POLARITY}/")
SAVE_DIR = Path(f"/weka/eickhoff/esx139/patch_analysis/ablation_results/allCat2Mean/gemma3_12b/{STYLE}/{POLARITY}/")
EMBEDS_DIR = Path(f"/weka/eickhoff/esx139/patch_analysis/mean_dataset_embeddings/all_cat2/gemma3_12b/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
INPAINT_STYLES = ["male_bg", "female_bg", "original"]

MODEL = "google/gemma-3-12b-it"
device = "cuda"

processor = AutoProcessor.from_pretrained(MODEL)
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
IMAGE_TOKEN_ID = model.config.image_token_id
config = model.config.to_dict()
image_size = config["vision_config"]["image_size"]       # 896
patch_size = config["vision_config"]["patch_size"]        # 14
num_tokens = config["mm_tokens_per_image"]                # 256

patches_per_side = image_size // patch_size               # 64
tokens_per_side = int(num_tokens ** 0.5)                  # 16
pool_size = patches_per_side // tokens_per_side           # 4

dataset = load_dataset("ucf-crcv/SB-Bench", split="real")
with open("/weka/eickhoff/esx139/inpainting/experiments/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

def run_experiment(sample_id, frac, inpaint_style, mean_embeds,):
    records = []

    if inpaint_style == "original":
        inputs, image_path = get_input(processor, dataset, id_to_index, sample_id, IMAGE_DIR, downsample_gemma3_12b,inpaited=False) #Change there to use inpainted
    else:
        inputs, image_path = get_input(processor, dataset, id_to_index, sample_id, IMAGE_DIR, downsample_gemma3_12b, inpaited=True, style=inpaint_style) #Change there to use inpainted 
    inputs = inputs.to(device)
    img = downsample_gemma3_12b(Image.open(image_path).convert("RGB"))
    orig_w, orig_h = img.size
    correct_answer = ["A", "B", "C"][dataset[id_to_index[sample_id]]["label"]]

    patch_map = build_gemma3_patch_to_pixel_map(
        input_ids=inputs["input_ids"][0],
        image_token_id= IMAGE_TOKEN_ID,
        original_height=orig_h,
        original_width=orig_w,
        image_size=image_size,
        patch_size=patch_size,
        mm_tokens_per_image=num_tokens,
    )
    all_image_positions = sorted(patch_map.keys())

    clean_outputs = model(**inputs)
    clean_probs = probs_tensor_to_dicts(choice_probs_ABC(clean_outputs, processor, inputs))[0]

    tokens_per_side = int(num_tokens ** 0.5)  # 16
    grid_h = tokens_per_side
    grid_w = tokens_per_side


    for row, col, tokens in sliding_window_on_grid(
        patch_map, frac=frac, mm_tokens_per_image=num_tokens
    ):
        replacements = {0: mean_embeds}
        ablated_outputs = ablated_forward(
            model, inputs,
            selected_positions=tokens,
            replacements=replacements,
            per_position=True,
            all_image_positions=all_image_positions,
        )
        ablated_probs = probs_tensor_to_dicts(choice_probs_ABC(ablated_outputs, processor, inputs))[0]

        records.append({
            "sample_id":       sample_id,
            "variant":         inpaint_style,
            "grid_h":          grid_h,
            "grid_w":          grid_w,
            "win_size":        max(2, round(min(grid_h, grid_w) * frac)),
            "frac":            frac,
            "stride":          max(1, max(2, round(min(grid_h, grid_w) * frac)) // 2),
            "win_row":         row,
            "win_col":         col,
            "clean_probs":     clean_probs,
            "ablated-probs":   ablated_probs,
            "correct_answer":  correct_answer,
        })

    return records

if __name__ == "__main__":
    frac = 0.33
    sample_ids = [d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]

    for inpaint_style in INPAINT_STYLES:
        baselines = torch.load(EMBEDS_DIR / f"original_mean_embeds.pt")
        mean_embeds = baselines["mean_embeds"]   # (256, 3840) — no mean_ds needed

        all_records = []
        for sample_id in sample_ids:
            records = run_experiment(sample_id, frac, inpaint_style, mean_embeds)
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        df.to_parquet(SAVE_DIR / f"ablation_results_{inpaint_style}.parquet")