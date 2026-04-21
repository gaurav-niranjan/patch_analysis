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
IMAGE_DIR = Path(f"/weka/eickhoff/esx139/flux_inpainting/flux_klein/inpainting_results/cat_2/F")
INPAINT_STYLES = ["original"]
SAVE_DIR = Path(f"/weka/eickhoff/esx139/patch_analysis/mean_dataset_embeddings/all_cat2/qwen8/{STYLE}/{POLARITY}/")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
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
NUM_DS_LAYERS = len(model.config.vision_config.deepstack_visual_indexes)

dataset = load_dataset("ucf-crcv/SB-Bench", split="real")
with open("/weka/eickhoff/esx139/inpainting/experiments/id_to_index.json", "r") as f:
    id_to_index = json.load(f)

def run_experiment(sample_id, inpaint_style):

    if inpaint_style == "original":
        inputs, image_path = get_input(processor, dataset, id_to_index, sample_id, IMAGE_DIR, downsample_qwen3_8b_p0, inpaited=False) #Change there to use inpainted
    else:
        inputs, image_path = get_input(processor, dataset, id_to_index, sample_id, IMAGE_DIR, downsample_qwen3_8b_p0, inpaited=True, style=inpaint_style) #Change there to use inpainted 
    inputs = inputs.to(device)
    
    image_embeds, deepstack_visual_embeds = get_deepstack_visual_embeds(model, inputs)

    return image_embeds, deepstack_visual_embeds
    

    

    

if __name__ == "__main__":
    sample_ids = [d.name for d in IMAGE_DIR.iterdir() if d.is_dir()]


    for inpaint_style in INPAINT_STYLES:
        save_path = SAVE_DIR / f"{inpaint_style}_mean_embeds.pt"

        #Running sums - constant memory
        embed_sum = None
        ds_sums = [None] * NUM_DS_LAYERS
        count = 0

        for idx, sample_id in enumerate(sample_ids):
            image_embeds, deepstack_visual_embeds = run_experiment(sample_id, inpaint_style)
            embeds_cpu = image_embeds.detach().float().cpu()
            ds_cpu = [ds.detach().float().cpu() for ds in deepstack_visual_embeds]

            if embed_sum is None:
                embed_sum = embeds_cpu
                ds_sums = [ds.clone() for ds in ds_cpu]
            else:
                assert embeds_cpu.shape == embed_sum.shape, \
                    f"Sample {idx}: {embeds_cpu.shape} != {embed_sum.shape}"
                embed_sum += embeds_cpu
                for i, ds in enumerate(ds_cpu):
                    assert ds.shape == ds_sums[i].shape, \
                        f"Sample {idx}, DS {i}: {ds.shape} != {ds_sums[i].shape}"
                    ds_sums[i] += ds
            count += 1

            if (idx + 1) % 100 == 0:
                print(f"  [{inpaint_style}] {idx+1}/{len(sample_ids)}")

        mean_embeds = embed_sum / count
        mean_ds = [ds_sum / count for ds_sum in ds_sums]
        torch.save({"mean_embeds": mean_embeds, "mean_ds": mean_ds}, save_path)
        print(f"Saved {save_path}  (n={count})")




