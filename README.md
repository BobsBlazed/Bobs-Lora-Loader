# Bobs LoRA Loader for ComfyUI

|||
|---|---|
|![image](https://github.com/user-attachments/assets/f614b579-c232-4f33-b994-f196c225edcf)|![image](https://github.com/user-attachments/assets/fca84c9b-211e-41fc-86a9-583e187cd6f1)|

An advanced LoRA loader for ComfyUI that provides granular, block-level control over how a LoRA is applied to both **SDXL** and **FLUX** models, giving you unparalleled control over your image generation process.

This node allows you to go beyond a single strength slider and specify different weights for distinct parts of the model, such as the text encoder, the U-Net input blocks, and the output blocks. This is particularly useful for mixing and matching LoRA concepts, strengthening character details while reducing stylistic influence, or vice-versa.

## Features

-   **Dual Model Support**: Separate, optimized loaders for `SDXL` and `FLUX` models, each tailored to the architecture's specific blocks.
-   **Granular Block-Level Control**: Fine-tune the strength of a LoRA on different conceptual parts of the diffusion model.
-   **Intelligent Presets**: Comes with pre-configured presets for common use cases like `Character`, `Style`, `Concept`, `Detail & Texture` and `Fix Hands/Anatomy`.
-   **Full Customization**: Set the `preset` to `Custom` to get direct slider control over every block for ultimate fine-tuning.
-   **Dialect-proof LoRA compatibility**: Classification runs on the *canonical model key* each patch targets, after ComfyUI has translated the LoRA's own naming scheme. Every format ComfyUI can load — kohya `lora_unet_*`, OneTrainer `lora_transformer_*`, diffusers `transformer.*`, LyCORIS, DiffSynth, PEFT — is bucketed correctly, including fused `qkv` / `linear1` patches.
-   **Geometry-aware FLUX blocks**: Block ranges are derived from the loaded model's own `depth` / `depth_single_blocks`, so FLUX.1 dev/schnell and pruned or distilled variants all map correctly instead of falling into "Other Tensors".
-   **Per-block report**: A third `info` output (and a matching console log) shows, per block, the weight used, how many tensors were found, and how many were actually patched.
-   **Optional CLIP**: leave the `clip` input unconnected to patch the model only.
-   **Standard LoRA Functionality**: To use it like a standard LoRA loader, simply select the `Full (Normal LoRA)` preset.

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

## How to Use

1.  In ComfyUI, add the node by right-clicking, selecting "Add Node," and navigating to the `Bobs_Nodes` category.
2.  Choose either **Bobs LoRA Loader (SDXL)** or **Bobs LoRA Loader (FLUX)** depending on your base model.
3.  Connect your `MODEL` and `CLIP` outputs into the corresponding inputs on the node. `CLIP` is optional — leave it unconnected to patch the model only.
4.  Select the LoRA you wish to apply from the `lora_name` dropdown.
5.  Use the `preset` dropdown to quickly apply a set of block weights for a specific purpose (e.g. "Character" to focus on subject detail).
6.  For maximum control, set the `preset` to `Custom` and adjust the individual block sliders.
7.  The main `strength` slider acts as a global multiplier for all other block weights, allowing you to scale the entire effect up or down.
8.  Hook the `info` output up to a preview-text node (or read the console) to see exactly which blocks the LoRA actually touched.

> **Note on presets:** a preset other than `Custom` *overrides* the sliders — it does not blend with them. Only `strength` still applies on top.

### Reading the `info` output

```
[FLUX] mylora.safetensors  (preset: Character)
block                                     weight   found  applied
Text Conditioning                           1.00       1        1
Early Downsampling (Composition)            0.60      16       16
Mid Upsampling (Detail Generation)          0.00      48        0
Text Encoder                                1.00       4        4
TOTAL                                                200      133
```

-   **found** — tensors in this LoRA that belong to that block.
-   **applied** — tensors actually patched. A block with `found > 0` and `applied 0` was skipped because its weight is `0.00`.
-   A block with `found 0` means the LoRA simply contains no weights for it; the console log spells this out.

## Block Layout

### FLUX

Ranges below are for the canonical FLUX.1 geometry (19 double-stream blocks, 38 single-stream blocks). Other depths are scaled proportionally.

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

| Block | Covers |
|---|---|
| Text Encoder | CLIP-L / CLIP-G |
| Input Blocks | `input_blocks.*` |
| Middle Block | `middle_block.*` |
| Output Blocks | `output_blocks.*` |
| Other Tensors | `time_embed`, `label_emb`, `out.*` |

## Why Use Block-Weighted LoRA?

A single LoRA file often contains training for multiple concepts (e.g. a character's face, their clothing, and the overall artistic style). A standard LoRA loader applies the LoRA with one uniform strength across the entire model.

This can be limiting. For example:
-   You might want a character's features but not the stiff, overbaked style it was trained with.
-   You might want a LoRA's artistic style but not the character concepts embedded within it.

By assigning different strengths to different model blocks, you can selectively emphasize or de-emphasize these aspects. The SDXL loader provides coarse control over the main UNet stages, while the FLUX loader offers even finer-grained control over conceptual phases like "Composition," "Refinement," and "Final Textures."

## Development

The block-classification logic lives in `bobs_blocks.py` and deliberately imports nothing from ComfyUI or torch, so it can be tested on a bare interpreter:

```bash
python -m unittest discover -s tests -v
```

## Changelog

### 1.1.0

**Block weighting now actually works.** This release fixes a defect that made the per-block sliders unreliable for every LoRA.

-   **Fixed: patches were classified by the wrong key.** ComfyUI's `key_map` maps a LoRA's key name to the *model state-dict key string* it targets (or a `(key, offset)` tuple for fused FLUX `qkv` / `linear1` weights). The previous code treated those values as `nn.Module` objects and indexed `[0]` on them, which on a string yields its first *character*. Every patch therefore resolved through a single arbitrary lookup, so all weights were effectively bucketed together rather than per block. Classification now runs directly on the canonical target key, and fused `(key, offset)` patches are unpacked correctly.
-   **Fixed: SDXL "unclassified" patches were silently discarded.** Anything that did not match input/middle/output/text-encoder was collected and then never applied. Those tensors now have their own `Other Tensors` weight.
-   **Fixed: model and CLIP patches are routed properly.** The FLUX loader previously pushed every patch group at its block strength into *both* the model and the CLIP patcher, with no separate text-encoder control. Model and text-encoder patches are now split by which key map owns the key, and FLUX gains a `Text Encoder` slider.
-   **Fixed: dead FLUX index ranges.** The old table mapped `double_blocks.19–28`, which do not exist on any FLUX model. Ranges are now computed from the loaded model's own `depth` / `depth_single_blocks`, so pruned and distilled variants map correctly too.
-   **Security: no more bare `torch.load`.** Non-safetensors LoRAs are read through `comfy.utils.load_torch_file(..., safe_load=True)`, which sets `weights_only=True`. Loading a `.ckpt`/`.pt` LoRA no longer risks executing pickled code.
-   **Added: LoRA file caching.** The file is re-read only when its path, size or mtime changes, instead of on every graph execution.
-   **Added: `info` string output** with a per-block table of weight / found / applied, plus console logging that explains empty blocks.
-   **Added: `comfy.lora_convert` support** (guarded), so BFL-control, Wan-Fun and USO LoRAs are converted before loading.
-   **Added: optional `clip` input**, tooltips on every widget, node descriptions, and a `Detail & Texture` preset for both families.
-   **Added: unit tests** and CI covering the classification and strength-resolution logic.
-   Errors (missing file, unreadable file, no matching keys) now return the graph inputs unchanged with an explanation on the `info` output instead of only logging.

**Compatibility:** existing workflows keep working. New widgets were appended after the existing ones and new outputs after the existing ones, so saved widget values and links stay aligned. Expect different — correct — results from the same slider settings, since the sliders previously did not target the blocks they named.
