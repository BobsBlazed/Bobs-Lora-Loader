# How classification works

This document explains the mechanism behind the three nodes, the reasoning
behind two non-obvious design choices, and how to add support for a new
architecture.

## The core idea: classify on the target key, not the LoRA key

A LoRA file's own key names vary wildly by exporter — kohya writes
`lora_unet_double_blocks_0_img_attn_qkv`, OneTrainer writes
`lora_transformer_...`, diffusers writes `transformer.transformer_blocks.0...`,
and LyCORIS, DiffSynth and PEFT each differ again.

ComfyUI already solves this. `comfy.lora.model_lora_keys_unet()` and
`model_lora_keys_clip()` build a **key map** from the *loaded model*, whose keys
are every dialect's spelling and whose values are the canonical model
state-dict key each one targets:

```python
key_map["lora_unet_double_blocks_0_img_attn_qkv"] = "diffusion_model.double_blocks.0.img_attn.qkv.weight"
key_map["transformer.transformer_blocks.0.attn.to_q"] = ("diffusion_model.double_blocks.0.img_attn.qkv.weight", (0, 0, 3072))
```

`comfy.lora.load_lora()` then returns a patch dict keyed by those *values*. So
this project never parses LoRA key names at all — it classifies the canonical
target. If ComfyUI can load the LoRA, we can bucket it, and new exporter formats
are free.

Two consequences worth knowing:

- A patch key is either a plain `str`, or a `(key, offset)` tuple when several
  LoRA tensors patch different slices of one fused weight (FLUX packs `q`, `k`
  and `v` into a single `qkv`). `_patch_target()` unwraps both. A real FLUX.1
  key map contains 534 such tuples, so this is the common case, not an edge one.
- Routing model vs. CLIP patches needs no name guessing: the UNet key map's
  values are the set of UNet-owned targets, and anything outside it goes to the
  text encoder.

## Design choice 1: stack matching uses the raw dotted key

Everywhere else, keys are normalised (`.` and `/` → `_`, lower-cased) so token
checks like `"_txt_in"` are simple substring tests. **Block-stack matching
deliberately does not use the normalised form.**

Normalising erases the separator that carries the structure. Consider an SDXL
key:

```
diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight
```

Normalised, it becomes `..._input_blocks_4_1_transformer_blocks_0_...`, where a
search for a stack named `blocks` or `transformer_blocks` happily matches the
*inner* one — the attention module's transformer blocks, not the UNet stage the
key actually belongs to. On the dotted key, `input_blocks.4` is unambiguously
the outermost component.

The same reasoning distinguishes a stack named `mystery_blocks` from one named
`blocks`: `mystery_blocks.0` and `blocks.0` are distinct, but
`_mystery_blocks_0` and `_blocks_0` both contain `_blocks_0`.

The FLUX and SDXL classifiers go one step further and anchor to the *start* of
the module path (after an optional `diffusion_model.` / `transformer.` prefix),
because anchoring on `.` alone is not enough — the nested stack is dot-preceded
too. FLUX's and SDXL's stacks are always top-level, so a nested
`transformer_blocks` can never be mistaken for a top-level one.

## Design choice 2: the Universal loader discovers the layout

ComfyUI ships close to a hundred model configs and adds more regularly, so a
per-family block table would be permanently out of date. Instead,
`bobs_universal.discover_layout()` scans the model's own key set for module path
components that look like a repeated stack — a name ending in `blocks`, `layers`
or `refiner`, followed by an index — and records each stack's size.

The discovered stacks are ordered by `STACK_ORDER` (a known execution order,
with unrecognised names sorted after it alphabetically) and concatenated. Every
block then has a position on a single 0..1 depth axis:

```
fraction = (offset_of_stack + index + 0.5) / total_blocks
```

which is bucketed into Early / Early-Mid / Mid / Late-Mid / Late. Because the
sizes come from the model, a pruned 8+16 FLUX and a full 19+38 FLUX both map
correctly, and the same five sliders mean the same thing on a 60-block
Qwen-Image as on a 20-stage SDXL UNet.

Keys outside any stack are matched against token lists for embeddings
(`Input & Embeddings`) and output heads (`Output Head`), falling through to
`Other Tensors`.

Negative indices are resolved against the stack size: ComfyUI's SD3 key map
addresses the final block as `joint_blocks.-1`.

## Strength resolution

```
Custom preset:  block_weight_slider × strength
Named preset:   preset_block_weight × preset_strength × strength
```

A named preset ignores the sliders entirely rather than blending with them.
`strength` always applies on top.

## Adding a new architecture

**Usually nothing is needed.** If the model's blocks live in a stack named
`*blocks` / `*layers` / `*refiner` with an index, the Universal loader finds it.
Check by loading a LoRA and reading the `info` output: it names the architecture
and the stacks discovered.

If tensors land in `Other Tensors`:

- **A stack was missed** — its name does not end in a recognised suffix. Extend
  `_STACK_RE` in `bobs_universal.py`, and add the name to `STACK_ORDER` if its
  execution position matters relative to other stacks.
- **Embedding or head tensors were missed** — add the token to `_INPUT_TOKENS`
  or `_OUTPUT_TOKENS`. Keep output tokens anchored: a bare substring like
  `_head_` also matches `multi_head_attention`.

To add a whole new *family* with its own named blocks (as FLUX and SDXL have),
call `bobs_blocks.register_family()` with the block list, presets, text-encoder
block name and tooltips, then add a node subclassing `_BobsLoraLoaderBase` with
`FAMILY`, `BLOCKS`, `CATCH_ALL_BLOCK` and a `_classifier()`. Registration is
explicit rather than done by module-level assignment, so behaviour never depends
on import order.

**Widget order is a compatibility contract.** ComfyUI serialises widget values
positionally, so reordering or removing an entry in a `ALL_*_BLOCKS` list
silently corrupts saved workflows. Only ever append. `tests/test_bobs_blocks.py`
pins the existing order.

## Testing

`bobs_blocks.py` and `bobs_universal.py` import nothing from ComfyUI or torch,
so the whole classification layer is testable on a bare interpreter:

```bash
python -m unittest discover -s tests -v
```

The suite covers both index-range tables, regex shadowing between nested and
top-level stacks, head-token vs. block-key precedence, full-state-dict sweeps
asserting nothing falls into `Other Tensors`, non-standard depths, preset and
slider precedence, the widget-order contract, and the node's registration,
report and mismatch warning.

Beyond that, the classifiers have been exercised against real ComfyUI: models
built from ComfyUI's own configs on torch's `meta` device (full module
structure, real key names, no weight memory) plus its `*_to_diffusers` key
tables, across SD1.5, SDXL, SD3, FLUX, AuraFlow, PixArt, LTX-Video, Lumina,
Qwen-Image and Wan. That harness lives outside the repository because it needs a
ComfyUI checkout and torch; the findings it produced are pinned as offline
regression tests in `tests/test_bobs_universal.py`.
