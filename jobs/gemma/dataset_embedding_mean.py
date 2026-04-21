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
IMAGE_DIR = Path(f"/weka/eickhoff/esx139/flux_inpainting/flux_klein/inpainting_results/cat_2/F")
INPAINT_STYLES = ["original"]
SAVE_DIR = Path(f"/weka/eickhoff/esx139/patch_analysis/mean_dataset_embeddings/all_cat2/gemma3_12b/{STYLE}/{POLARITY}/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
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

def run_experiment(sample_id, inpaint_style):

    if inpaint_style == "original":
        inputs, image_path = get_input(processor, dataset, id_to_index, sample_id, IMAGE_DIR, downsample_gemma3_12b, inpaited=False) #Change there to use inpainted
    else:
        inputs, image_path = get_input(processor, dataset, id_to_index, sample_id, IMAGE_DIR, downsample_gemma3_12b, inpaited=True, style=inpaint_style) #Change there to use inpainted 
    inputs = inputs.to(device)
    
    image_embeds, _ = get_image_embeds(model, inputs, image_token_id=IMAGE_TOKEN_ID)
    #print(f"Sample {sample_id}, {inpaint_style}: Image embeds shape: {image_embeds.shape}")

    return image_embeds
    

    

    

if __name__ == "__main__":
    sample_ids = [d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]

    for inpaint_style in INPAINT_STYLES:
        save_path = SAVE_DIR / f"{inpaint_style}_mean_embeds.pt"

        running_sum = None
        count = 0

        for idx, sample_id in enumerate(sample_ids):
            image_embeds = run_experiment(sample_id, inpaint_style)
            embeds_cpu = image_embeds.detach().float().cpu()

            if running_sum is None:
                running_sum = embeds_cpu
            else:
                assert embeds_cpu.shape == running_sum.shape, \
                    f"Sample {idx}: {embeds_cpu.shape} != {running_sum.shape}"
                running_sum += embeds_cpu
            count += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{inpaint_style}] {idx+1}/{len(sample_ids)}")

        mean_embeds = running_sum / count
        torch.save({"mean_embeds": mean_embeds}, save_path)
        print(f"Saved {save_path}  (n={count})")


