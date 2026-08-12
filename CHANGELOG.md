# Changelog

All notable changes to Bobs LoRA Loader. Versions match the [Comfy Registry](https://registry.comfy.org/) releases.

## 1.2.1

Code-review fixes. No behaviour change for a correctly-paired LoRA and model.

-   **The FLUX and SDXL classifiers now anchor on the module path**, matching
    the Universal loader. They previously matched on an underscore-normalised
    key, which erases the separator that tells an *outermost* block stack from
    one nested inside a block. An SDXL key such as
    `input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight` was read as
    double-stream block 0, so pointing the FLUX node at an SDXL model filed
    every tensor under "Early Downsampling". Real FLUX and SDXL keys are
    unaffected — this only ever misfired across families.
-   **Mismatched loader and model now warn** instead of failing silently. If
    90% or more of a LoRA's UNet tensors land in one block, the `info` output
    and the log say so and point at the Universal loader. Measured: a real SDXL
    key set through the FLUX node warns at 99%, a FLUX key set through the SDXL
    node at 100%, while all six correctly-paired combinations stay silent.
-   **Families register explicitly** via `register_family()` rather than by
    mutating `bobs_blocks`'s tables at import time, removing an import-order
    dependency that also made the test suite fragile.
-   **Tooltips are per family.** FLUX/SDXL and Universal each define a block
    called "Text Encoder"; the Universal wording used to overwrite the other.
-   **`Output Head` detection tightened.** The `_head_` substring also matched
    `multi_head_attention` and `head_dim_proj`; it is now anchored to a real
    head module.
-   An unknown `preset` name (only reachable from hand-edited workflow JSON)
    logs a warning instead of silently behaving as `Custom`.
-   Test suite grown to 71, including a new `tests/test_node_behaviour.py`
    covering node registration, widget order, the report and the new warning.

## 1.2.0

**New: a Universal loader covering every architecture ComfyUI supports.**

-   **`Bobs LoRA Loader (Universal)`** handles SD1.5, SD2, SDXL, SD3/3.5, FLUX,
    Chroma, AuraFlow, PixArt, HiDream, Qwen-Image, Wan, LTX-Video, Mochi,
    HunyuanVideo/DiT, Lumina, Cosmos and anything else built as stacks of
    repeated blocks. The block layout is *discovered from the loaded model* —
    stack names, stack sizes and their execution order — rather than read from a
    per-family table, so pruned, distilled and brand-new architectures work
    without a code change.
-   Weights are assigned along a normalised depth axis (Early → Late), plus
    embeddings, output head and text encoder, so the same sliders mean the same
    thing across very different models.
-   The `info` output now reports the detected architecture and the discovered
    stacks, e.g. `architecture: QwenImage  transformer_blocks[60]  (total 60)`.
    The FLUX and SDXL loaders report their architecture too.

**Verified against real ComfyUI.** The classification logic was run against
models built from ComfyUI's own configs and against its `*_to_diffusers` key
tables, covering SD1.5, SDXL, SD3, FLUX (full and pruned geometry), AuraFlow,
PixArt, LTX-Video, Lumina, Qwen-Image and Wan — 8,000+ authentic state-dict
keys, all classified with none falling through to `Other Tensors`. That pass
found and fixed several real gaps that synthetic fixtures had missed:

-   SD3 addresses its final block as `joint_blocks.-1`; negative indices are now
    resolved against the stack size instead of failing to match.
-   Lumina's `noise_refiner` / `context_refiner` stacks are recognised and
    ordered ahead of the main `layers` stack.
-   AuraFlow's native `double_layers` / `single_layers` names are handled, not
    just the diffusers spelling.
-   Conditioning embedders that previously fell through — PixArt's `ar_embedder`,
    `csize_embedder` and `t_block`, LTX-Video's `adaln_single` and
    `scale_shift_table`, Qwen-Image's `txt_norm`, Wan's `time_projection`,
    AuraFlow's `cond_seq_linear` / `positional_encoding` — now land in
    `Input & Embeddings`.
-   FLUX's ControlNet `pos_embed_input` now maps to `Image Hint`.

Existing FLUX and SDXL workflows are unaffected: no widget was added, removed or
reordered on those two nodes.

## 1.1.0

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
