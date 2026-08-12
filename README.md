# Bobs LoRA Loader for ComfyUI

|||
|---|---|
|![image](https://github.com/user-attachments/assets/f614b579-c232-4f33-b994-f196c225edcf)|![image](https://github.com/user-attachments/assets/fca84c9b-211e-41fc-86a9-583e187cd6f1)|

Block-weighted LoRA loading for ComfyUI. Instead of one strength slider for the
whole model, you get a slider per conceptual part of it — text encoder,
composition, subject, style, detail, texture — so you can keep the parts of a
LoRA you want and turn down the parts you don't.

Three nodes, all under the `Bobs_Nodes` category:

| Node | Use it for |
|---|---|
| **Bobs LoRA Loader (FLUX)** | FLUX.1 dev/schnell, Chroma and other FLUX variants, with FLUX's named blocks |
| **Bobs LoRA Loader (SDXL)** | SDXL and SD1.5/SD2, with the UNet's input/middle/output stages |
| **Bobs LoRA Loader (Universal)** | Everything else, and anything new — see [supported architectures](#universal) |

## Features

-   **Works on any supported architecture.** The Universal loader discovers the
    model's block layout at runtime rather than reading a hard-coded table, so
    pruned, distilled and brand-new architectures work without an update.
-   **Granular block-level control.** Tune a LoRA's strength separately on each
    conceptual part of the diffusion model.
-   **Presets** for common jobs: `Character`, `Style`, `Concept`,
    `Detail & Texture`, `Fix Hands/Anatomy` — plus `Custom` for the sliders.
-   **Dialect-proof compatibility.** Classification runs on the *canonical model
    key* each patch targets, after ComfyUI has translated the LoRA's own naming
    scheme. Every format ComfyUI can load — kohya `lora_unet_*`, OneTrainer
    `lora_transformer_*`, diffusers `transformer.*`, LyCORIS, DiffSynth, PEFT —
    is bucketed correctly, including fused `qkv` / `linear1` patches.
-   **Per-block report.** An `info` output (and a matching console log) shows the
    detected architecture and, per block, the weight used, how many tensors were
    found and how many were actually patched.
-   **Tells you when the pairing is wrong.** Point a node at the wrong model
    family and it says so instead of silently doing nothing useful.
-   **Optional CLIP.** Leave the `clip` input unconnected to patch the model only.
-   **Drop-in standard behaviour.** Select `Full (Normal LoRA)` to get the same
    result as ComfyUI's built-in `LoraLoader`.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone https://github.com/BobsBlazed/Bobs-Lora-Loader
    ```
3.  Restart ComfyUI.

Or install **Bobs_LoRA_Loader** from the ComfyUI Manager / Comfy Registry.

## How to Use

1.  Right-click → **Add Node** → `Bobs_Nodes`, and pick the node matching your
    base model (see the table above). When in doubt, use **Universal** — it
    works on FLUX and SDXL too, just with depth-based block names instead of
    architecture-specific ones.
2.  Connect `MODEL` and `CLIP`. `CLIP` is optional — leave it unconnected to
    patch the model only.
3.  Pick the LoRA from `lora_name`.
4.  Choose a `preset`, or set it to `Custom` and drive the sliders yourself.
5.  `strength` is a global multiplier applied on top of every block weight.
6.  Hook `info` up to a preview-text node (or read the console) to see exactly
    which blocks the LoRA touched.

> **Presets override the sliders.** Anything other than `Custom` ignores the
> slider values entirely — it does not blend with them. Only `strength` still
> applies on top.

### Reading the `info` output

```
[UNIVERSAL] mylora.safetensors  (preset: Style)
architecture: QwenImage  transformer_blocks[60]  (total 60)
block                                     weight   found  applied
Text Encoder                                0.20       0        0
Input & Embeddings                          1.00       9        9
Early Blocks (Composition)                  0.10     384      384
Early-Mid Blocks (Subject)                  0.00     384        0
Mid Blocks (Concept & Style)                0.50     384      384
Late-Mid Blocks (Detail)                    1.00     384      384
Late Blocks (Texture)                       1.00     384      384
Output Head                                 1.00       4        4
Other Tensors                               1.00       0        0
TOTAL                                               1933     1549
```

-   **architecture** — the backbone ComfyUI detected, and the block stacks found
    in it. The FLUX and SDXL nodes report their architecture here too.
-   **found** — tensors in this LoRA belonging to that block.
-   **applied** — tensors actually patched. `found > 0` with `applied 0` means
    the block's weight is `0.00`, which is usually what you asked for.
-   `found 0` means the LoRA contains no weights for that block at all; the
    console log spells this out per block.
-   **Other Tensors** should normally be `0` or close to it. A large number here
    means the classifier could not place those tensors — see
    [Troubleshooting](#troubleshooting).

## Block Layout

### FLUX

Ranges below are for the canonical FLUX.1 geometry (19 double-stream blocks, 38
single-stream blocks). Other depths are scaled proportionally.

| Block | Covers |
|---|---|
| Text Encoder | CLIP-L / T5 text encoder weights |
| Text Conditioning | `txt_in` |
| Timestep Embedding | `time_in` |
| Image Hint | `img_in` |
| Guidance Embedding | `guidance_in` |
| Vector Embedding | `vector_in` |
| Early Downsampling (Composition) | `double_blocks.0–3` |
| Mid Downsampling (Subject & Concept) | `double_blocks.4–7` |
| Late Downsampling (Refinement) | `double_blocks.8–9` |
| Core/Middle Block (Style Focus) | `double_blocks.10–18`, `single_blocks.0–7` |
| Early Upsampling (Initial Style) | `single_blocks.8–15` |
| Mid Upsampling (Detail Generation) | `single_blocks.16–31` |
| Late Upsampling (Final Textures) | `single_blocks.32–37` |
| Final Output Layer (Latent Projection) | `final_layer` |
| Other Tensors | anything unmatched (normally empty) |

### SDXL

Also works for SD1.5 and SD2 — same UNet shape, different depth.

| Block | Covers |
|---|---|
| Text Encoder | CLIP-L / CLIP-G |
| Input Blocks | `input_blocks.*` |
| Middle Block | `middle_block.*` |
| Output Blocks | `output_blocks.*` |
| Other Tensors | `time_embed`, `label_emb`, `out.*` |

### Universal

The Universal loader works on a single normalised **depth axis**. Almost every
diffusion backbone is one or more ordered stacks of repeated blocks:

```
UNet (SD1.5 / SDXL)     input_blocks.N -> middle_block -> output_blocks.N
Dual-stream DiT (FLUX)  double_blocks.N -> single_blocks.N
HiDream                 double_stream_blocks.N -> single_stream_blocks.N
MMDiT (SD3)             joint_blocks.N
AuraFlow                double_layers.N -> single_layers.N
Qwen-Image / LTX-Video  transformer_blocks.N
Wan / Mochi / PixArt    blocks.N
Lumina                  noise_refiner.N -> context_refiner.N -> layers.N
```

Those stacks are **discovered from the loaded model**, concatenated in execution
order, and every block gets a position from 0 to 1 along the result. That axis is
split into five buckets, so the same five sliders mean the same thing on a
19+38-block FLUX, a 60-block Qwen-Image and a 20-stage SDXL UNet.

| Block | Covers |
|---|---|
| Text Encoder | Every text-encoder weight (CLIP / T5 / LLM) |
| Input & Embeddings | Patch, timestep, guidance and context embedders |
| Early Blocks (Composition) | First 20% of the stack |
| Early-Mid Blocks (Subject) | 20–40% |
| Mid Blocks (Concept & Style) | 40–60% |
| Late-Mid Blocks (Detail) | 60–80% |
| Late Blocks (Texture) | Final 20% |
| Output Head | Final projection back to latent space |
| Other Tensors | Anything unmatched (normally empty) |

Directly verified against SD1.5, SDXL, SD3, FLUX (full and pruned geometry),
AuraFlow, PixArt, LTX-Video, Lumina, Qwen-Image and Wan — every state-dict key
of each classified, none falling through to `Other Tensors`.

Architectures such as HiDream, Chroma, Mochi, HunyuanVideo and Cosmos are
supported by the same mechanism but were not part of that run, so treat them as
expected-to-work rather than confirmed. Because the layout comes from the model
rather than a table, an architecture missing from both lists will usually still
work — the `info` output tells you whether the stacks were found.

Use the dedicated FLUX or SDXL loader when you want that architecture's named
blocks; use Universal for everything else, or when you want one node whose
sliders behave consistently across models.

## Why Use Block-Weighted LoRA?

A single LoRA file often contains training for multiple concepts — a character's
face, their clothing, and the overall artistic style. A standard LoRA loader
applies all of it at one uniform strength.

That can be limiting:

-   You might want a character's features but not the stiff, overbaked style it
    was trained with.
-   You might want a LoRA's artistic style but not the character baked into it.
-   Two LoRAs might fight each other when both are applied at full strength.

Roughly, earlier blocks carry composition and subject identity while later
blocks carry style, detail and texture — so a `Character` preset keeps the early
blocks and drops the late ones, and `Style` does the reverse. The exact split is
in the tables above.

## Troubleshooting

**The nodes don't appear in the menu.** Check the ComfyUI startup console for a
traceback mentioning `bobs_`. The package needs no dependencies beyond ComfyUI
itself, so this is usually a partial clone or a stale `__pycache__`.

**`WARNING: N% of UNet tensors landed in '<block>'`.** The node and the model
disagree about the architecture — e.g. an SDXL model in the FLUX node. Switch to
the node matching your model, or to Universal. The LoRA still applies, but the
block sliders won't mean what their names say.

**`none of its tensors match this model`.** The LoRA was trained for a different
architecture than the loaded model. Nothing is applied and the model passes
through untouched.

**High `Other Tensors` count.** The classifier placed those tensors nowhere.
For the FLUX/SDXL nodes this usually means the wrong node for the model; try
Universal. If Universal also shows a high count, that's worth
[an issue](https://github.com/BobsBlazed/Bobs-Lora-Loader/issues) — please
include the `info` output, which names the architecture and stacks it found.

**Results differ from the built-in `LoraLoader`.** With `Full (Normal LoRA)` at
the same strength they should match. Any other preset deliberately differs —
that's the point of the node.

## Development

The classification logic imports nothing from ComfyUI or torch, so the test
suite runs on a bare interpreter:

```bash
python -m unittest discover -s tests -v
```

| File | Contents |
|---|---|
| `bobs_blocks.py` | FLUX and SDXL block tables, presets, classifiers, strength resolution |
| `bobs_universal.py` | Runtime stack discovery and the depth-axis classifier |
| `bobs_lora_loader.py` | The ComfyUI nodes: key maps, patch routing, reporting |
| `tests/` | 71 tests, no ComfyUI or torch required |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how classification works
and how to add support for a new architecture.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md). Latest release: **1.2.1**.

## License

[Apache-2.0](LICENSE)
