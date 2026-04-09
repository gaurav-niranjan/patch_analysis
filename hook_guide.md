Let me trace through the exact sequence. The decoder loop looks like this:

```python
hidden_states = inputs_embeds  # main visual features already placed here

for layer_idx, decoder_layer in enumerate(self.layers):
    # ──── PRE-HOOK fires here ────
    hidden_states = decoder_layer(hidden_states, ...)
    # ──── deepstack injection ────
    if layer_idx in range(len(deepstack_visual_embeds)):
        hidden_states[image_positions] += deepstack[layer_idx]
```

Now let's walk through what each pre-hook captures:

```
═══════════════════════════════════════════════════════════════

PRE-HOOK LAYER 0 captures:
  → inputs_embeds with main features placed at image positions
  → Operations done: NONE. Just the raw projected ViT final layer output.
  
  This is: embed(text_tokens) + projected_vit_output(image_tokens)

───────────────────────────────────────────────────────────────
  Layer 0 runs:        attention + FFN across ALL tokens
  deepstack[0] added:  image positions += ViT layer 5 features
───────────────────────────────────────────────────────────────

PRE-HOOK LAYER 1 captures:
  → layer_0_output + deepstack[0]
  → Operations done on image tokens:
      1. Self-attention in layer 0 (mixed with all text tokens)
      2. FFN in layer 0
      3. deepstack[0] addition (ViT layer 5 features)

───────────────────────────────────────────────────────────────
  Layer 1 runs:        attention + FFN across ALL tokens
  deepstack[1] added:  image positions += ViT layer 11 features
───────────────────────────────────────────────────────────────

PRE-HOOK LAYER 2 captures:
  → layer_1_output + deepstack[1]
  → Operations done on image tokens:
      1. Everything from above
      2. Self-attention in layer 1
      3. FFN in layer 1
      4. deepstack[1] addition (ViT layer 11 features)

───────────────────────────────────────────────────────────────
  Layer 2 runs:        attention + FFN across ALL tokens
  deepstack[2] added:  image positions += ViT layer 17 features
───────────────────────────────────────────────────────────────

PRE-HOOK LAYER 3 captures:
  → layer_2_output + deepstack[2]
  → Operations done on image tokens:
      1. Everything from above
      2. Self-attention in layer 2
      3. FFN in layer 2
      4. deepstack[2] addition (ViT layer 17 features)

  This is the LAST point where new vision info was injected.
  After this, no more visual information enters.

═══════════════════════════════════════════════════════════════
```

The key thing to notice: **only pre-hook 0 is pure.** Pre-hooks 1, 2, 3 capture states that are a mix of transformer processing AND deepstack additions. You can't cleanly separate "what the transformer did" from "what deepstack added" at those points — they're entangled through attention.

So when you compute the mean from the clean capture for replacement, be aware of what you're averaging:

```
Layer 0 mean: mean of pure projected visual features
Layer 1 mean: mean of (attention-mixed features + deepstack[0])
Layer 2 mean: mean of (2 layers of attention + deepstack[0] + deepstack[1])
Layer 3 mean: mean of (3 layers of attention + deepstack[0] + deepstack[1] + deepstack[2])
```

This is actually what you want — you're replacing with "what the model normally sees at this point," which includes all the natural processing that happened before.


## For the deepstack injections, should i ablate the tokens on the deepstack or the tokens on the hidden state after the deepstack is applied?

**Ablating the DeepStack features themselves** (before they get added to the hidden state) only removes the *new* fine-grained visual detail being injected at that layer. But the hidden state at that position still carries whatever information propagated through previous LLM layers via self-attention. Even though you ablated at input, neighboring patches share information through self-attention in layers 1 and 2, so by layer 3 the "ablated" position may have partially recovered patch information from its neighbors. The DeepStack ablation doesn't touch that recovered signal.

**Ablating the hidden state after injection** wipes *everything* at that position — both the propagated residual stream and the freshly injected DeepStack features. This is a full reset of that spatial position at that layer.

**For your purpose — "which patches matter for the answer" — ablate the hidden state after injection.** You want to completely sever information flow from that patch. If you only ablate the DeepStack contribution, you're leaving a leaky channel through self-attention, and your attribution map will underestimate how much the model relies on those patches.

**However**, ablating only the DeepStack features is a valid and interesting *separate* experiment. It answers: "how much does DeepStack's fine-grained injection contribute beyond what the LLM can figure out from the input tokens alone?" That could be a nice complementary analysis in a paper — but it's a different research question from the one you started with.

So: **hidden state after injection for your main analysis, DeepStack features only as an optional secondary experiment.**

## I am thinking about what would it mean to replace the selected token values with the mean of all image tokens, in that layer (initilial or the one with the deepstack) during a clean run.

Good to think about this carefully. Here's what it means:

**At the input layer:** Each vision token carries information specific to its spatial patch — texture, color, edges, object parts at that location. The mean across all 300 tokens is a representation that averages over *all* spatial locations. It retains information that is globally shared across the image (overall brightness, dominant color palette, maybe "this is an indoor scene") but loses anything spatially specific (there's a face *here*, a motorcycle *there*). So replacing a patch with the mean is saying: "this position still looks like a generic patch from *this image*, but it no longer knows what was specifically at its location."

**At a DeepStack layer:** The same logic applies but now the representations are richer. After several layers of self-attention, each hidden state has already integrated contextual information from other positions. The mean at this layer represents something like "the average contextualized visual representation across all spatial positions." It still carries image-level semantics but has lost position-specific information. Replacing here says: "reset this position to a generic, contextually averaged state at this depth of processing."

**The key insight is what you're preserving vs. destroying:**

You preserve the overall distributional properties — the replacement is in-distribution, the LLM doesn't see anything "weird." You also preserve image-level information since the mean is image-specific. What you destroy is the *spatially specific* information: what makes patch (3,7) different from patch (12,15).

**This is exactly why it's a good intervention for your question.** You're asking "does the model need to know what's *specifically here* to answer correctly?" The mean keeps everything else normal and only removes the spatial specificity.

**One nuance worth noting:** if your image has very distinct regions (like a person vs. a dark background), the mean will be somewhere "between" those — it won't perfectly represent either. That's fine. It's uninformative about location, which is the point.

Want to move forward to implementation now?

