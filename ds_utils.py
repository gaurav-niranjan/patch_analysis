from pathlib import Path
from datasets import load_dataset
import json
from PIL import Image

def get_id_to_indexMap():
    try:
        with open("/weka/eickhoff/esx139/inpainting/experiments/id_to_index.json", "r") as f:
            id_to_index = json.load(f)
        return id_to_index
    
    except FileNotFoundError:
        raise FileNotFoundError("id_to_index.json not found. Building id_to_index map from dataset.")
    
def get_genderMap():
    try:
        with open("/weka/eickhoff/esx139/polarity/gender_polarity/gender_assignments.json", "r") as f:
            gender_map = json.load(f)
        id_to_gender_map = {
            item["id"]: {
                "A": item.get("ans0_gender"),
                "B": item.get("ans1_gender"),
                "C": item.get("ans2_gender"),
                }
                for item in gender_map
            }
        return id_to_gender_map

    except FileNotFoundError:
        raise FileNotFoundError("gender_assignments.json not found.")

