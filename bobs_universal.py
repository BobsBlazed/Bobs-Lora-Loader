"""
Architecture-agnostic block analysis for Bobs LoRA Loader.

The FLUX and SDXL loaders in :mod:`bobs_blocks` know their architecture's block
names up front. That does not scale: ComfyUI ships close to a hundred model
configs, and new ones land regularly. Rather than maintain a table per family,
this module *discovers* a model's block layout from the model's own key set at
runtime, then buckets each key by how deep it sits in the execution order.

The observation that makes this work: essentially every diffusion backbone is
one or more ordered stacks of repeated blocks, whatever they are named —

    UNet (SD1.5 / SDXL)    input_blocks.N  -> middle_block -> output_blocks.N
    Dual-stream DiT (FLUX) double_blocks.N -> single_blocks.N
    HiDream                double_stream_blocks.N -> single_stream_blocks.N
    MMDiT (SD3)            joint_blocks.N
    AuraFlow               joint_transformer_blocks.N -> single_transformer_blocks.N
    Qwen-Image / LTXV      transformer_blocks.N
    Wan / Mochi            blocks.N
    Lumina                 layers.N

Concatenating those stacks in execution order gives every block a position on a
single 0..1 depth axis, which is then split into five buckets. Anything outside
a stack is an embedding/input layer, an output head, or genuinely unrecognised.

Because the stack sizes come from the model rather than a constant, this adapts
to pruned, distilled and brand-new architectures without a code change.

Imports nothing from ComfyUI or torch; see ``tests/test_bobs_universal.py``.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .bobs_blocks import (
    ALL_BLOCKS,
    BLOCK_TOOLTIPS,
    LORA_BLOCK_PRESETS,
    TEXT_ENCODER_BLOCK,
    normalize_key,
)

# -----------------------------------------------------------------------------#
#                                 BLOCK NAMES                                  #
# -----------------------------------------------------------------------------#
#
# Widget values serialise positionally — only ever append to this list.

UNIVERSAL_TEXT_ENCODER = "Text Encoder"
UNIVERSAL_INPUT = "Input & Embeddings"
UNIVERSAL_EARLY = "Early Blocks (Composition)"
UNIVERSAL_EARLY_MID = "Early-Mid Blocks (Subject)"
UNIVERSAL_MID = "Mid Blocks (Concept & Style)"
UNIVERSAL_LATE_MID = "Late-Mid Blocks (Detail)"
UNIVERSAL_LATE = "Late Blocks (Texture)"
UNIVERSAL_OUTPUT = "Output Head"
UNIVERSAL_OTHER = "Other Tensors"

ALL_UNIVERSAL_BLOCKS: List[str] = [
    UNIVERSAL_TEXT_ENCODER,
    UNIVERSAL_INPUT,
    UNIVERSAL_EARLY,
    UNIVERSAL_EARLY_MID,
    UNIVERSAL_MID,
    UNIVERSAL_LATE_MID,
    UNIVERSAL_LATE,
    UNIVERSAL_OUTPUT,
    UNIVERSAL_OTHER,
]

# Depth buckets, as upper-exclusive fractions of the concatenated block stacks.
_DEPTH_BUCKETS: Sequence[Tuple[float, str]] = (
    (0.2, UNIVERSAL_EARLY),
    (0.4, UNIVERSAL_EARLY_MID),
    (0.6, UNIVERSAL_MID),
    (0.8, UNIVERSAL_LATE_MID),
    (1.01, UNIVERSAL_LATE),
)

UNIVERSAL_TOOLTIPS: Dict[str, str] = {
    UNIVERSAL_TEXT_ENCODER: "All text-encoder weights (CLIP / T5 / LLM), whatever the model uses.",
    UNIVERSAL_INPUT: "Patch, timestep, guidance and context embedding layers feeding the block stack.",
    UNIVERSAL_EARLY: "First 20% of the block stack. Global composition and layout.",
    UNIVERSAL_EARLY_MID: "20–40% of the block stack. Subject identity.",
    UNIVERSAL_MID: "40–60% of the block stack. Concept and dominant style.",
    UNIVERSAL_LATE_MID: "60–80% of the block stack. Detail generation.",
    UNIVERSAL_LATE: "Final 20% of the block stack. Fine texture and surface.",
    UNIVERSAL_OUTPUT: "Final projection back to latent space.",
    UNIVERSAL_OTHER: "Tensors that matched no stack, embedding or head pattern.",
}

# -----------------------------------------------------------------------------#
#                              STACK  DISCOVERY                                #
# -----------------------------------------------------------------------------#

# Execution order of the stack names we know about. A model only ever uses a
# few of these; the order here is what puts them on one axis correctly.
# Unknown stack names still work — they sort after these, alphabetically.
STACK_ORDER: Tuple[str, ...] = (
    # UNet
    "input_blocks",
    "middle_block",
    "output_blocks",
    # Dual-stream DiT: FLUX, HunyuanVideo, Chroma
    "double_blocks",
    "single_blocks",
    # HiDream
    "double_stream_blocks",
    "single_stream_blocks",
    # MMDiT: SD3 / SD3.5
    "joint_blocks",
    # Lumina: input refiners run ahead of the main stack
    "noise_refiner",
    "context_refiner",
    # AuraFlow (ComfyUI names them *_layers; the diffusers names also appear)
    "double_layers",
    "single_layers",
    "joint_transformer_blocks",
    "single_transformer_blocks",
    # Qwen-Image, LTX-Video, PixArt, Cosmos
    "transformer_blocks",
    # Wan, Mochi, Kandinsky
    "blocks",
    # Lumina, generic transformer stacks
    "layers",
)

# Stack matching runs on the RAW dotted key, never the underscore-normalised
# one. Normalising would erase the separator that tells "mystery_blocks.0"
# (stack "mystery_blocks") apart from a stack literally named "blocks" — both
# collapse to "..._blocks_0". The dot boundary is the whole signal.
#
# No list of known names is needed to *match*: a repeated stack is always a
# module path component ending in "blocks"/"layers" followed by an index.
# STACK_ORDER below is used only to order the stacks once discovered.
# The name prefix is optional so a stack named exactly "blocks" or "layers"
# (Wan, Mochi, PixArt, Lumina) matches just as well as "single_stream_blocks".
# "refiner" is included for Lumina's noise_refiner/context_refiner stacks.
# The index may be negative: ComfyUI's SD3 key map addresses the final joint
# block as "joint_blocks.-1", which is resolved against the stack size below.
_STACK_RE = re.compile(
    r"(?:^|\.)((?:[A-Za-z][A-Za-z0-9_]*)?(?:blocks|layers|refiner))\.(-?\d+)(?=\.|$)",
    re.IGNORECASE,
)
# ``middle_block`` is a single module rather than an indexed list.
_MIDDLE_RE = re.compile(r"(?:^|\.)(middle_block)(?=\.|$)", re.IGNORECASE)

# Tokens that identify a non-stack tensor. Checked in order, output before
# input, because some head names also contain an embedding-ish word.
_OUTPUT_TOKENS: Tuple[str, ...] = (
    "_final_layer",
    "_final_linear",
    "_proj_out",
    "_norm_out",
    "_unpatchify",
    "_final_norm",
    "_head_",
)
_OUTPUT_RE = re.compile(r"_out_\d+(?=_|$)")  # SD "out.0" / "out.2" head

_INPUT_TOKENS: Tuple[str, ...] = (
    "_img_in",
    "_txt_in",
    "_x_embedder",
    "_patch_embed",
    "_patchify",
    "_time_in",
    "_time_embed",
    "_time_text_embed",
    "_timestep_embed",
    "_t_embedder",
    "_guidance_in",
    "_vector_in",
    "_context_embedder",
    "_caption_projection",
    "_y_embedder",
    "_label_emb",
    "_pos_embed",
    "_text_embedding",
    "_condition_embedder",
    "_img_emb",
    "_register_tokens",
    "_cap_embedder",
    "_input_hint",
    "_init_x_linear",       # AuraFlow
    "_cond_seq_linear",     # AuraFlow text conditioning
    "_positional_encoding",
    "_modf",                # AuraFlow final modulation
    "_adaln_single",        # LTX-Video / PixArt shared modulation
    "_t_block",             # PixArt timestep modulation
    "_scale_shift_table",
    "_txt_norm",            # Qwen-Image text stream norm
    "_time_projection",     # Wan
    "_rope",
    "_freqs",
    # Generic catch-all, last: covers *_embedder / *_embedding / *_embed names
    # we have not enumerated (PixArt's ar_embedder and csize_embedder, and
    # whatever the next architecture calls its conditioning embedders).
    "_embed",
)


def _stack_match(model_key: str) -> Optional[Tuple[str, int]]:
    """Return ``(stack_name, index)`` for the outermost stack in a raw key.

    The *leftmost* match is deliberate, and matters twice over:

    - an SDXL key ``input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight``
      belongs to ``input_blocks``, not the attention module's inner
      ``transformer_blocks``;
    - ``middle_block.1.transformer_blocks.0...`` belongs to ``middle_block``,
      which is why the two patterns are compared by position rather than
      tried in a fixed order.
    """
    stack = _STACK_RE.search(model_key)
    middle = _MIDDLE_RE.search(model_key)

    if stack and middle:
        return ((stack.group(1), int(stack.group(2))) if stack.start() < middle.start()
                else ("middle_block", 0))
    if stack:
        return stack.group(1), int(stack.group(2))
    if middle:
        return "middle_block", 0
    return None


def _stack_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (STACK_ORDER.index(name), "")
    except ValueError:
        return (len(STACK_ORDER), name)


class BlockLayout:
    """The block-stack layout discovered from one model's key set."""

    def __init__(self, stacks: Dict[str, int]):
        #: stack name -> number of blocks, in execution order
        self.stacks: Dict[str, int] = {
            name: stacks[name] for name in sorted(stacks, key=_stack_sort_key)
        }
        self.offsets: Dict[str, int] = {}
        running = 0
        for name, size in self.stacks.items():
            self.offsets[name] = running
            running += size
        self.total: int = running

    def __bool__(self) -> bool:
        return self.total > 0

    def describe(self) -> str:
        if not self.stacks:
            return "no block stacks detected"
        parts = [f"{name}[{size}]" for name, size in self.stacks.items()]
        return " -> ".join(parts) + f"  (total {self.total})"

    def depth_fraction(self, stack: str, index: int) -> Optional[float]:
        """Position of a block on the 0..1 depth axis, centred in its slot."""
        if not self.total or stack not in self.offsets:
            return None
        size = self.stacks[stack]
        if index < 0:  # Python-style: "joint_blocks.-1" is the last block.
            index += size
        # Guard against an index beyond what discovery saw.
        index = min(max(index, 0), max(size - 1, 0))
        return (self.offsets[stack] + index + 0.5) / self.total


def discover_layout(model_keys: Iterable[str]) -> BlockLayout:
    """Infer the block-stack layout from a model's state-dict key names."""
    sizes: Dict[str, int] = {}
    for key in model_keys:
        found = _stack_match(key)
        if found is None:
            continue
        name, index = found
        if index + 1 > sizes.get(name, 0):
            sizes[name] = index + 1
    return BlockLayout(sizes)


# -----------------------------------------------------------------------------#
#                               CLASSIFICATION                                 #
# -----------------------------------------------------------------------------#

def classify_universal_key(model_key: str, layout: BlockLayout) -> str:
    """Bucket a UNet/DiT state-dict key using a discovered :class:`BlockLayout`."""
    nk = normalize_key(model_key)

    found = _stack_match(model_key)
    if found is not None:
        fraction = layout.depth_fraction(*found)
        if fraction is not None:
            for upper, name in _DEPTH_BUCKETS:
                if fraction < upper:
                    return name
            return UNIVERSAL_LATE

    for token in _OUTPUT_TOKENS:
        if token in nk:
            return UNIVERSAL_OUTPUT
    if _OUTPUT_RE.search(nk):
        return UNIVERSAL_OUTPUT

    for token in _INPUT_TOKENS:
        if token in nk:
            return UNIVERSAL_INPUT

    return UNIVERSAL_OTHER


# -----------------------------------------------------------------------------#
#                                  PRESETS                                     #
# -----------------------------------------------------------------------------#
#
# Registered into the shared tables in bobs_blocks so resolve_block_strengths
# treats "UNIVERSAL" exactly like the two hand-tuned families.

UNIVERSAL_PRESETS = {
    "Custom": {},
    "Full (Normal LoRA)": {
        "strength": 1.0,
        "block_weights": {name: 1.0 for name in ALL_UNIVERSAL_BLOCKS},
    },
    "Character": {
        "strength": 1.0,
        "block_weights": {
            UNIVERSAL_TEXT_ENCODER: 1.0,
            UNIVERSAL_INPUT: 1.0,
            UNIVERSAL_EARLY: 0.8,
            UNIVERSAL_EARLY_MID: 1.0,
            UNIVERSAL_MID: 1.0,
            UNIVERSAL_LATE_MID: 0.2,
            UNIVERSAL_LATE: 0.0,
            UNIVERSAL_OUTPUT: 0.0,
            UNIVERSAL_OTHER: 1.0,
        },
    },
    "Style": {
        "strength": 1.0,
        "block_weights": {
            UNIVERSAL_TEXT_ENCODER: 0.2,
            UNIVERSAL_INPUT: 1.0,
            UNIVERSAL_EARLY: 0.1,
            UNIVERSAL_EARLY_MID: 0.0,
            UNIVERSAL_MID: 0.5,
            UNIVERSAL_LATE_MID: 1.0,
            UNIVERSAL_LATE: 1.0,
            UNIVERSAL_OUTPUT: 1.0,
            UNIVERSAL_OTHER: 1.0,
        },
    },
    "Concept": {
        "strength": 1.0,
        "block_weights": {
            UNIVERSAL_TEXT_ENCODER: 1.0,
            UNIVERSAL_INPUT: 1.0,
            UNIVERSAL_EARLY: 1.0,
            UNIVERSAL_EARLY_MID: 0.9,
            UNIVERSAL_MID: 0.7,
            UNIVERSAL_LATE_MID: 0.4,
            UNIVERSAL_LATE: 0.2,
            UNIVERSAL_OUTPUT: 0.0,
            UNIVERSAL_OTHER: 1.0,
        },
    },
    "Detail & Texture": {
        "strength": 1.0,
        "block_weights": {
            UNIVERSAL_TEXT_ENCODER: 0.0,
            UNIVERSAL_INPUT: 1.0,
            UNIVERSAL_EARLY: 0.0,
            UNIVERSAL_EARLY_MID: 0.0,
            UNIVERSAL_MID: 0.2,
            UNIVERSAL_LATE_MID: 1.0,
            UNIVERSAL_LATE: 1.0,
            UNIVERSAL_OUTPUT: 1.0,
            UNIVERSAL_OTHER: 0.0,
        },
    },
    "Fix Hands/Anatomy": {
        "strength": 0.4,
        "block_weights": {
            UNIVERSAL_TEXT_ENCODER: 0.2,
            UNIVERSAL_INPUT: 1.0,
            UNIVERSAL_EARLY: 1.0,
            UNIVERSAL_EARLY_MID: 0.5,
            UNIVERSAL_MID: 0.0,
            UNIVERSAL_LATE_MID: 0.0,
            UNIVERSAL_LATE: 0.0,
            UNIVERSAL_OUTPUT: 0.0,
            UNIVERSAL_OTHER: 0.0,
        },
    },
}

LORA_BLOCK_PRESETS["UNIVERSAL"] = UNIVERSAL_PRESETS
ALL_BLOCKS["UNIVERSAL"] = ALL_UNIVERSAL_BLOCKS
TEXT_ENCODER_BLOCK["UNIVERSAL"] = UNIVERSAL_TEXT_ENCODER
BLOCK_TOOLTIPS.update(UNIVERSAL_TOOLTIPS)
