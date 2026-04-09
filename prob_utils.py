import torch
import torch.nn.functional as F
import math

def last_real_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    attention_mask: [B, L] with 1 for real tokens, 0 for pad
    returns last index of a real token for each sample: [B]
    Works for left or right padding.
    returns index of the last real token in the input seq.
    """
    B, L = attention_mask.shape
    rev = attention_mask.flip(dims=[1])                
    last = (L - 1) - rev.long().argmax(dim=1) 
    return last


@torch.no_grad()
def next_token_probs(outputs, inputs):
    logits = outputs.logits  # [B, L, V]

    # Index of the last non-pad token in each sequence
    last_idx = last_real_token_index(inputs["attention_mask"]) #[B]
    next_logits = logits[torch.arange(logits.size(0), device=logits.device), last_idx, :]  # [B, V]
    return F.softmax(next_logits, dim=-1)  # [B, V]

def _single_token_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None # in case of multi-token, return None. E.g, "A)" maps to two tokens.

@torch.no_grad()
def choice_probs_ABC(outputs, processor, inputs):
    tok = processor.tokenizer
    probs = next_token_probs(outputs, inputs)  # [B, V]

    #To score all kinds of variants
    variants = {
        "A": ["A", " A", "\nA", "A)", "A."],
        "B": ["B", " B", "\nB", "B)", "B."],
        "C": ["C", " C", "\nC", "C)", "C."],
    }

    ids = {}
    for k, vs in variants.items():
        ids[k] = [tid for v in vs if (tid := _single_token_id(tok, v)) is not None]

    sA = probs[:, ids["A"]].sum(dim=1) if len(ids["A"]) else torch.zeros(probs.size(0), device=probs.device) #If none of the answer candidates are single-token.
    sB = probs[:, ids["B"]].sum(dim=1) if len(ids["B"]) else torch.zeros(probs.size(0), device=probs.device) #[B,1]
    sC = probs[:, ids["C"]].sum(dim=1) if len(ids["C"]) else torch.zeros(probs.size(0), device=probs.device)

    scores = torch.stack([sA, sB, sC], dim=1) #[B,3]

    Z = scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    probs_abc = scores / Z

    return probs_abc

def probs_tensor_to_dicts(probs_abc):
    labels = ["A", "B", "C"]

    # if tensor -> convert once
    if torch.is_tensor(probs_abc):
        probs_abc = probs_abc.float().cpu().tolist()

    return [{labels[j]: probs_abc[i][j] for j in range(3)}
            for i in range(len(probs_abc))]


def entropy(probs_dict):
    ps = list(probs_dict.values())
    return -sum(p * math.log(p + 1e-12) for p in ps)

def confidence(probs_dict):
    return max(probs_dict.values())

