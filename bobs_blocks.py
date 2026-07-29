"""
Block definitions, presets and key-classification logic for Bobs LoRA Loader.

This module deliberately imports nothing from ComfyUI or torch so the mapping
logic can be exercised standalone (see ``tests/test_bobs_blocks.py``).

Everything here classifies *model state dict keys* — i.e. the values stored in
ComfyUI's ``key_map`` (``diffusion_model.double_blocks.3.img_attn.qkv.weight``,
``clip_l.transformer.text_model.encoder.layers.0.self_attn.k_proj.weight``, ...)
— not the raw key names found inside a LoRA file. ComfyUI has already done the
work of translating every exporter dialect (kohya, OneTrainer, diffusers,
lycoris, DiffSynth, ...) into these canonical names, so classifying on them is
both simpler and dialect-proof.
"""

import re
from typing import Dict, List, Sequence, Tuple

# -----------------------------------------------------------------------------#
#                                 BLOCK NAMES                                  #
# -----------------------------------------------------------------------------#
#
# NOTE ON ORDERING: ComfyUI serialises widget values positionally, so the order
# of these lists is part of the node's saved-workflow contract. Never reorder or
# remove an entry — only append new ones at the end.

FLUX_TEXT_CONDITIONING = "Text Conditioning"
FLUX_TIMESTEP = "Timestep Embedding"
FLUX_IMAGE_HINT = "Image Hint"
FLUX_GUIDANCE = "Guidance Embedding"
FLUX_VECTOR = "Vector Embedding"
FLUX_EARLY_DOWN = "Early Downsampling (Composition)"
FLUX_MID_DOWN = "Mid Downsampling (Subject & Concept)"
FLUX_LATE_DOWN = "Late Downsampling (Refinement)"
FLUX_CORE = "Core/Middle Block (Style Focus)"
FLUX_EARLY_UP = "Early Upsampling (Initial Style)"
FLUX_MID_UP = "Mid Upsampling (Detail Generation)"
FLUX_LATE_UP = "Late Upsampling (Final Textures)"
FLUX_FINAL = "Final Output Layer (Latent Projection)"
FLUX_OTHER = "Other Tensors"
FLUX_TEXT_ENCODER = "Text Encoder"  # appended in 1.1.0 — keep last

ALL_FLUX_BLOCKS: List[str] = [
    FLUX_TEXT_CONDITIONING,
    FLUX_TIMESTEP,
    FLUX_IMAGE_HINT,
    FLUX_GUIDANCE,
    FLUX_VECTOR,
    FLUX_EARLY_DOWN,
    FLUX_MID_DOWN,
    FLUX_LATE_DOWN,
    FLUX_CORE,
    FLUX_EARLY_UP,
    FLUX_MID_UP,
    FLUX_LATE_UP,
    FLUX_FINAL,
    FLUX_OTHER,
    FLUX_TEXT_ENCODER,
]

SDXL_TEXT_ENCODER = "Text Encoder"
SDXL_INPUT_BLOCKS = "Input Blocks"
SDXL_MIDDLE_BLOCK = "Middle Block"
SDXL_OUTPUT_BLOCKS = "Output Blocks"
SDXL_OTHER = "Other Tensors"  # appended in 1.1.0 — keep last

ALL_SDXL_BLOCKS: List[str] = [
    SDXL_TEXT_ENCODER,
    SDXL_INPUT_BLOCKS,
    SDXL_MIDDLE_BLOCK,
    SDXL_OUTPUT_BLOCKS,
    SDXL_OTHER,
]

# The single block per family that receives CLIP / text-encoder patches.
TEXT_ENCODER_BLOCK = {"FLUX": FLUX_TEXT_ENCODER, "SDXL": SDXL_TEXT_ENCODER}

BLOCK_TOOLTIPS: Dict[str, str] = {
    FLUX_TEXT_ENCODER: "CLIP-L / T5 text encoder weights. Lower this to keep a LoRA's look while weakening its trigger words.",
    FLUX_TEXT_CONDITIONING: "diffusion_model.txt_in — projects text embeddings into the transformer.",
    FLUX_TIMESTEP: "diffusion_model.time_in — timestep embedding.",
    FLUX_IMAGE_HINT: "diffusion_model.img_in — latent patch embedding.",
    FLUX_GUIDANCE: "diffusion_model.guidance_in — distilled guidance embedding.",
    FLUX_VECTOR: "diffusion_model.vector_in — pooled CLIP vector embedding.",
    FLUX_EARLY_DOWN: "First fifth of the double-stream blocks. Global composition and pose.",
    FLUX_MID_DOWN: "Second fifth of the double-stream blocks. Subject identity and concept.",
    FLUX_LATE_DOWN: "Tail of the first half of the double-stream blocks. Structural refinement.",
    FLUX_CORE: "Late double-stream plus early single-stream blocks. Dominant style carrier.",
    FLUX_EARLY_UP: "Early-middle single-stream blocks. Broad stylistic treatment.",
    FLUX_MID_UP: "Middle single-stream blocks. Detail generation.",
    FLUX_LATE_UP: "Final single-stream blocks. Fine texture, grain and skin detail.",
    FLUX_FINAL: "diffusion_model.final_layer — projection back to latent space.",
    FLUX_OTHER: "Any UNet tensor that did not match a block above. Normally near zero.",
    SDXL_INPUT_BLOCKS: "UNet down path (diffusion_model.input_blocks.*). Composition and structure.",
    SDXL_MIDDLE_BLOCK: "UNet bottleneck (diffusion_model.middle_block.*). Concept and style core.",
    SDXL_OUTPUT_BLOCKS: "UNet up path (diffusion_model.output_blocks.*). Style, detail and texture.",
    SDXL_OTHER: "UNet tensors outside the three stages (time_embed, label_emb, out.*).",
}
BLOCK_TOOLTIPS[SDXL_TEXT_ENCODER] = BLOCK_TOOLTIPS[FLUX_TEXT_ENCODER]

#: Per-family tooltip overrides, keyed family -> block -> text. Families share
#: some block *names* ("Text Encoder", "Other Tensors") but want different help
#: text, so a single flat dict cannot hold both.
FAMILY_TOOLTIPS: Dict[str, Dict[str, str]] = {}

# -----------------------------------------------------------------------------#
#                             PRESET  DEFINITIONS                              #
# -----------------------------------------------------------------------------#
#
# ``strength`` is a preset-level multiplier stacked on top of the node's global
# strength; ``block_weights`` is the per-block multiplier. A block missing from
# ``block_weights`` falls back to 1.0.
#
# Preset names are serialised by value, so appending new presets is safe.

LORA_BLOCK_PRESETS: Dict[str, Dict[str, Dict]] = {
    "FLUX": {
        "Custom": {},
        "Full (Normal LoRA)": {
            "strength": 1.0,
            "block_weights": {name: 1.0 for name in ALL_FLUX_BLOCKS},
        },
        "Character": {
            "strength": 1.0,
            "block_weights": {
                FLUX_TEXT_ENCODER: 1.0,
                FLUX_TEXT_CONDITIONING: 1.0,
                FLUX_TIMESTEP: 1.0,
                FLUX_IMAGE_HINT: 1.0,
                FLUX_GUIDANCE: 1.0,
                FLUX_VECTOR: 1.0,
                FLUX_EARLY_DOWN: 0.6,
                FLUX_MID_DOWN: 1.0,
                FLUX_LATE_DOWN: 0.4,
                FLUX_CORE: 1.0,
                FLUX_EARLY_UP: 0.1,
                FLUX_MID_UP: 0.0,
                FLUX_LATE_UP: 0.0,
                FLUX_FINAL: 0.0,
                FLUX_OTHER: 1.0,
            },
        },
        "Style": {
            "strength": 1.0,
            "block_weights": {
                FLUX_TEXT_ENCODER: 0.2,
                FLUX_TEXT_CONDITIONING: 0.2,
                FLUX_TIMESTEP: 1.0,
                FLUX_IMAGE_HINT: 1.0,
                FLUX_GUIDANCE: 1.0,
                FLUX_VECTOR: 1.0,
                FLUX_EARLY_DOWN: 0.1,
                FLUX_MID_DOWN: 0.0,
                FLUX_LATE_DOWN: 0.2,
                FLUX_CORE: 0.5,
                FLUX_EARLY_UP: 1.0,
                FLUX_MID_UP: 1.0,
                FLUX_LATE_UP: 1.0,
                FLUX_FINAL: 1.0,
                FLUX_OTHER: 1.0,
            },
        },
        "Concept": {
            "strength": 1.0,
            "block_weights": {
                FLUX_TEXT_ENCODER: 1.0,
                FLUX_TEXT_CONDITIONING: 1.0,
                FLUX_TIMESTEP: 1.0,
                FLUX_IMAGE_HINT: 1.0,
                FLUX_GUIDANCE: 1.0,
                FLUX_VECTOR: 1.0,
                FLUX_EARLY_DOWN: 1.0,
                FLUX_MID_DOWN: 0.9,
                FLUX_LATE_DOWN: 0.6,
                FLUX_CORE: 0.7,
                FLUX_EARLY_UP: 0.5,
                FLUX_MID_UP: 0.3,
                FLUX_LATE_UP: 0.1,
                FLUX_FINAL: 0.0,
                FLUX_OTHER: 1.0,
            },
        },
        "Detail & Texture": {
            "strength": 1.0,
            "block_weights": {
                FLUX_TEXT_ENCODER: 0.0,
                FLUX_TEXT_CONDITIONING: 0.0,
                FLUX_TIMESTEP: 1.0,
                FLUX_IMAGE_HINT: 1.0,
                FLUX_GUIDANCE: 1.0,
                FLUX_VECTOR: 1.0,
                FLUX_EARLY_DOWN: 0.0,
                FLUX_MID_DOWN: 0.0,
                FLUX_LATE_DOWN: 0.0,
                FLUX_CORE: 0.2,
                FLUX_EARLY_UP: 0.5,
                FLUX_MID_UP: 1.0,
                FLUX_LATE_UP: 1.0,
                FLUX_FINAL: 1.0,
                FLUX_OTHER: 0.0,
            },
        },
        "Fix Hands/Anatomy": {
            "strength": 0.4,
            "block_weights": {
                FLUX_TEXT_ENCODER: 0.0,
                FLUX_TEXT_CONDITIONING: 0.2,
                FLUX_TIMESTEP: 1.0,
                FLUX_IMAGE_HINT: 1.0,
                FLUX_GUIDANCE: 1.0,
                FLUX_VECTOR: 1.0,
                FLUX_EARLY_DOWN: 1.0,
                FLUX_MID_DOWN: 0.3,
                FLUX_LATE_DOWN: 0.0,
                FLUX_CORE: 0.0,
                FLUX_EARLY_UP: 0.0,
                FLUX_MID_UP: 0.0,
                FLUX_LATE_UP: 0.0,
                FLUX_FINAL: 0.0,
                FLUX_OTHER: 0.0,
            },
        },
    },

    "SDXL": {
        "Custom": {},
        "Full (Normal LoRA)": {
            "strength": 1.0,
            "block_weights": {b: 1.0 for b in ALL_SDXL_BLOCKS},
        },
        "Character": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 1.0,
                SDXL_INPUT_BLOCKS: 1.0,
                SDXL_MIDDLE_BLOCK: 1.0,
                SDXL_OUTPUT_BLOCKS: 0.2,
                SDXL_OTHER: 1.0,
            },
        },
        "Style": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.0,
                SDXL_INPUT_BLOCKS: 0.2,
                SDXL_MIDDLE_BLOCK: 0.5,
                SDXL_OUTPUT_BLOCKS: 1.0,
                SDXL_OTHER: 1.0,
            },
        },
        "Concept": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 1.0,
                SDXL_INPUT_BLOCKS: 0.8,
                SDXL_MIDDLE_BLOCK: 0.7,
                SDXL_OUTPUT_BLOCKS: 0.5,
                SDXL_OTHER: 1.0,
            },
        },
        "Detail & Texture": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.0,
                SDXL_INPUT_BLOCKS: 0.0,
                SDXL_MIDDLE_BLOCK: 0.2,
                SDXL_OUTPUT_BLOCKS: 1.0,
                SDXL_OTHER: 0.0,
            },
        },
        "Fix Hands/Anatomy": {
            "strength": 0.4,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.2,
                SDXL_INPUT_BLOCKS: 1.0,
                SDXL_MIDDLE_BLOCK: 0.4,
                SDXL_OUTPUT_BLOCKS: 0.0,
                SDXL_OTHER: 0.0,
            },
        },
    },
}

ALL_BLOCKS = {"FLUX": ALL_FLUX_BLOCKS, "SDXL": ALL_SDXL_BLOCKS}

#: Families defined in this module. Additional families register themselves via
#: :func:`register_family`; keep this constant for code that needs the built-in
#: two regardless of what else has been imported.
BUILTIN_FAMILIES = ("FLUX", "SDXL")


def register_family(family: str,
                    blocks: List[str],
                    presets: Dict[str, Dict],
                    text_encoder_block: str,
                    tooltips: Dict[str, str] = None) -> None:
    """Register an additional block family (see :mod:`bobs_universal`).

    Kept explicit rather than done by assignment at import time: the tables
    below are module state, and a silent import-order dependency makes both
    behaviour and tests fragile. Tooltips are stored per family so that two
    families sharing a block name (both call one "Text Encoder") do not
    overwrite each other's help text.
    """
    ALL_BLOCKS[family] = blocks
    LORA_BLOCK_PRESETS[family] = presets
    TEXT_ENCODER_BLOCK[family] = text_encoder_block
    FAMILY_TOOLTIPS[family] = dict(tooltips or {})


def tooltip_for(family: str, block: str) -> str:
    """Help text for a block, preferring the family's own wording."""
    return FAMILY_TOOLTIPS.get(family, {}).get(block) or BLOCK_TOOLTIPS.get(block, block)

# -----------------------------------------------------------------------------#
#                              KEY NORMALISATION                               #
# -----------------------------------------------------------------------------#

_SEPARATORS = re.compile(r"[./:]+")
_REPEATED_US = re.compile(r"_{2,}")


def normalize_key(key: str) -> str:
    """Lower-case a key and collapse ``. / :`` into single underscores.

    A leading underscore is always present in the result so that head-token
    checks such as ``"_txt_in"`` also match a key that *starts* with the token.
    """
    k = _SEPARATORS.sub("_", key).lower()
    k = _REPEATED_US.sub("_", k)
    return k if k.startswith("_") else "_" + k


# -----------------------------------------------------------------------------#
#                             FLUX  CLASSIFICATION                             #
# -----------------------------------------------------------------------------#

# Canonical FLUX.1 geometry: 19 double-stream blocks, 38 single-stream blocks.
FLUX_DEFAULT_DEPTH = 19
FLUX_DEFAULT_DEPTH_SINGLE = 38

# Fractional split points across each stack. The fractions are chosen so the
# canonical geometry reproduces the index ranges this node has always used,
# while non-standard depths (Flex, Chroma, pruned/distilled FLUX variants)
# scale proportionally instead of falling out into "Other Tensors".
_DOUBLE_SPLITS: Sequence[Tuple[float, str]] = (
    (4 / 19, FLUX_EARLY_DOWN),
    (8 / 19, FLUX_MID_DOWN),
    (10 / 19, FLUX_LATE_DOWN),
    (1.0, FLUX_CORE),
)
_SINGLE_SPLITS: Sequence[Tuple[float, str]] = (
    (8 / 38, FLUX_CORE),
    (16 / 38, FLUX_EARLY_UP),
    (32 / 38, FLUX_MID_UP),
    (1.0, FLUX_LATE_UP),
)

# Stack matching runs on the RAW dotted key, never the underscore-normalised
# one. Normalising erases the separator that distinguishes an *outermost* stack
# from one nested inside a block: an SDXL key
# "input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight" normalises to
# "..._input_blocks_4_1_transformer_blocks_0_..." where a naive search matches
# the inner transformer_blocks and reads the key as double-stream block 0.
# Anchoring on "." keeps the two apart, and makes the leftmost match the
# outermost stack.
#
# ``single`` is still tested before ``double`` so that
# "single_transformer_blocks.3" cannot be read as "transformer_blocks.3".
#
# Anchoring on "." alone is not enough — the nested transformer_blocks is also
# dot-preceded. FLUX's stacks are always *top level*, so the pattern is anchored
# to the start of the module path (after an optional known prefix). SDXL's
# nested stack then cannot match, because "input_blocks" sits in front of it.
_PREFIX = r"^(?:model\.)?(?:diffusion_model\.|transformer\.)?"
_RE_SINGLE_BLOCK = re.compile(
    _PREFIX + r"(?:single_blocks|single_transformer_blocks)\.(-?\d+)(?=\.|$)")
_RE_DOUBLE_BLOCK = re.compile(
    _PREFIX + r"(?:double_blocks|transformer_blocks)\.(-?\d+)(?=\.|$)")

# Head / tail tokens, most specific first.
_FLUX_TOKEN_MAP: Sequence[Tuple[str, str]] = (
    ("_time_text_embed_guidance_embedder", FLUX_GUIDANCE),
    ("_time_text_embed_text_embedder", FLUX_VECTOR),
    ("_time_text_embed_timestep_embedder", FLUX_TIMESTEP),
    ("_time_text_embed", FLUX_TIMESTEP),
    ("_guidance_in", FLUX_GUIDANCE),
    ("_vector_in", FLUX_VECTOR),
    ("_time_in", FLUX_TIMESTEP),
    ("_txt_in", FLUX_TEXT_CONDITIONING),
    ("_context_embedder", FLUX_TEXT_CONDITIONING),   # diffusers name for txt_in
    ("_img_in", FLUX_IMAGE_HINT),
    ("_x_embedder", FLUX_IMAGE_HINT),                # diffusers name for img_in
    ("_pos_embed_input", FLUX_IMAGE_HINT),           # FLUX ControlNet hint input
    ("_final_layer", FLUX_FINAL),
    ("_proj_out", FLUX_FINAL),                       # diffusers name for final_layer
    ("_norm_out", FLUX_FINAL),
)


def _stack_boundaries(splits: Sequence[Tuple[float, str]], depth: int) -> List[Tuple[int, str]]:
    """Turn fractional split points into ``(upper_exclusive_index, block)`` pairs."""
    boundaries: List[Tuple[int, str]] = []
    previous = 0
    for fraction, name in splits:
        upper = max(int(round(fraction * depth)), previous)
        boundaries.append((upper, name))
        previous = upper
    if boundaries:
        # The last bucket always absorbs the tail, whatever rounding did.
        boundaries[-1] = (depth, boundaries[-1][1])
    return boundaries


def flux_block_ranges(depth: int = FLUX_DEFAULT_DEPTH,
                      depth_single: int = FLUX_DEFAULT_DEPTH_SINGLE):
    """Return ``(double_boundaries, single_boundaries)`` for a FLUX geometry."""
    return (_stack_boundaries(_DOUBLE_SPLITS, max(int(depth), 0)),
            _stack_boundaries(_SINGLE_SPLITS, max(int(depth_single), 0)))


def _bucket(index: int, boundaries: Sequence[Tuple[int, str]], fallback: str) -> str:
    for upper, name in boundaries:
        if index < upper:
            return name
    return fallback


def _resolve_index(index: int, boundaries: Sequence[Tuple[int, str]]) -> int:
    """Resolve a Python-style negative block index against the stack size."""
    if index >= 0:
        return index
    size = boundaries[-1][0] if boundaries else 0
    return max(index + size, 0)


def classify_flux_key(model_key: str, ranges=None) -> str:
    """Map a FLUX UNet state-dict key to one of :data:`ALL_FLUX_BLOCKS`."""
    double_bounds, single_bounds = ranges or flux_block_ranges()

    # Block stacks match on the raw dotted key (see _RE_SINGLE_BLOCK); the
    # token checks below match on the normalised one.
    match = _RE_SINGLE_BLOCK.search(model_key)
    if match:
        return _bucket(_resolve_index(int(match.group(1)), single_bounds),
                       single_bounds, FLUX_LATE_UP)

    match = _RE_DOUBLE_BLOCK.search(model_key)
    if match:
        return _bucket(_resolve_index(int(match.group(1)), double_bounds),
                       double_bounds, FLUX_CORE)

    nk = normalize_key(model_key)
    for token, block in _FLUX_TOKEN_MAP:
        if token in nk:
            return block

    return FLUX_OTHER


# -----------------------------------------------------------------------------#
#                             SDXL  CLASSIFICATION                             #
# -----------------------------------------------------------------------------#

# Anchored on "." for the same reason as the FLUX patterns above: the stage is
# the outermost path component, not whatever is nested inside a block.
_RE_SDXL_STAGE = re.compile(
    _PREFIX + r"(input_blocks|middle_block|output_blocks)(?=\.|$)")

_SDXL_STAGE_BLOCKS = {
    "input_blocks": SDXL_INPUT_BLOCKS,
    "middle_block": SDXL_MIDDLE_BLOCK,
    "output_blocks": SDXL_OUTPUT_BLOCKS,
}


def classify_sdxl_key(model_key: str) -> str:
    """Map an SDXL UNet state-dict key to one of :data:`ALL_SDXL_BLOCKS`."""
    match = _RE_SDXL_STAGE.search(model_key)
    if match:
        return _SDXL_STAGE_BLOCKS[match.group(1)]
    return SDXL_OTHER


# -----------------------------------------------------------------------------#
#                              STRENGTH RESOLUTION                             #
# -----------------------------------------------------------------------------#

def resolve_block_strengths(family: str,
                            preset: str,
                            strength: float,
                            overrides: Dict[str, float]) -> Dict[str, float]:
    """Combine the global strength, the preset and the per-block sliders.

    ``Custom`` (and any unknown preset name) uses the slider values; every other
    preset ignores the sliders and uses its own table, scaled by the preset's
    own strength multiplier and the node's global strength.
    """
    names = ALL_BLOCKS[family]
    config = LORA_BLOCK_PRESETS[family].get(preset) or {}
    weights = config.get("block_weights")

    if not weights:
        return {name: float(overrides.get(name, 1.0)) * float(strength) for name in names}

    base = float(strength) * float(config.get("strength", 1.0))
    return {name: float(weights.get(name, 1.0)) * base for name in names}
