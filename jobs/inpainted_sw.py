import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from pathlib import Path
from datasets import load_dataset
import json
from PIL import Image
from patch_analysis.utils import *
from patch_analysis.prob_utils import choice_probs_ABC, probs_tensor_to_dicts
from patch_analysis.hook_utils import *
import pandas as pd

STYLE = "style0"
POLARITY = "polarity0"
IMAGE_DIR = Path(f"/weka/eickhoff/esx139/flux_inpainting/flux_klein/consistent_set/qwen8/{STYLE}/cat_2/{POLARITY}/")
SAVE_DIR = Path(f"/weka/eickhoff/esx139/patch_analysis/ablation_results/{STYLE}/{POLARITY}/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
INPAINT_STYLES = ["original"]  #["female_bg", "male_bg"]

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
device = "cuda"

processor = AutoProcessor.from_pretrained(MODEL)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
IMAGE_TOKEN_ID = model.config.image_token_id
MERGE_SIZE = processor.image_processor.merge_size
PATCH_SIZE = processor.image_processor.patch_size

ds = load_dataset("ucf-crcv/SB-Bench", split="real")
with open("/weka/eickhoff/esx139/inpainting/experiments/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

def run_experiment(sample_id, frac, inpaint_style):
    records = []

    if inpaint_style == "original":
        inputs, image_path = get_input(processor, ds, id_to_index, sample_id, IMAGE_DIR, inpaited=False) #Change there to use inpainted
    else:
        inputs, image_path = get_input(processor, ds, id_to_index, sample_id, IMAGE_DIR, inpaited=True, style=inpaint_style) #Change there to use inpainted 
    image_grid_thw = inputs['image_grid_thw']
    inputs = inputs.to(device)
    img = downsample_image(Image.open(image_path).convert("RGB"))
    orig_w, orig_h = img.size
    correct_answer = ["A", "B", "C"][ds[id_to_index[sample_id]]["label"]]

    patch_map = build_qwen3vl_patch_to_pixel_map(
        input_ids=inputs["input_ids"][0],
        image_token_id=IMAGE_TOKEN_ID,
        image_grid_thw=image_grid_thw,
        processor=processor,
        original_height=orig_h,
        original_width=orig_w,
    )
    all_image_positions = sorted(patch_map.keys())

    captured, clean_outputs = clean_capture(model, inputs, all_image_positions, generate=False)
    clean_probs = probs_tensor_to_dicts(choice_probs_ABC(clean_outputs, processor, inputs))[0]

    text_inputs = get_textonly_input(processor, ds, id_to_index, sample_id)
    text_only_outputs = model(**text_inputs.to(device))
    text_only_probs = probs_tensor_to_dicts(choice_probs_ABC(text_only_outputs, processor, text_inputs))[0]

    mean_replacements = compute_replacements(captured, strategy="mean")

    t, h_pre, w_pre = image_grid_thw[0].tolist()
    grid_h = h_pre // MERGE_SIZE
    grid_w = w_pre // MERGE_SIZE

    for row, col, tokens in sliding_window_on_grid(
        patch_map, image_grid_thw, frac=frac, merge_size=MERGE_SIZE
    ):
        ablated_outputs = ablated_forward(
            model, inputs,
            selected_positions=tokens,
            replacements=mean_replacements,
            generate=False,
        )
        ablated_probs = probs_tensor_to_dicts(choice_probs_ABC(ablated_outputs, processor, inputs))[0]

        records.append({
            "sample_id":       sample_id,
            "variant":         inpaint_style,
            "grid_h":          grid_h,
            "grid_w":          grid_w,
            "win_size":        len(set(range(row, row + round(min(grid_h, grid_w) * frac)))),
            "frac":            frac,
            "stride":          max(1, max(2, round(min(grid_h, grid_w) * frac)) // 2),
            "win_row":         row,
            "win_col":         col,
            "clean_probs":     clean_probs,
            "text-only_probs": text_only_probs,
            "ablated-probs":   ablated_probs,
            "correct_answer":  correct_answer,
        })

    return records

if __name__ == "__main__":
    frac = 0.33
    sample_ids = [d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]

    for inpaint_style in INPAINT_STYLES:
        all_records = []
        for sample_id in sample_ids:
            records = run_experiment(sample_id, frac, inpaint_style)
            all_records.extend(records)

        df = pd.DataFrame(all_records)
        df.to_parquet(SAVE_DIR / f"ablation_results_{inpaint_style}.parquet")