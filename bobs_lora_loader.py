import logging
from typing import Dict, Any, List, Tuple
import os
import re

import comfy.utils
import comfy.lora
import folder_paths
import torch
from safetensors.torch import load_file as safe_load_file

# -----------------------------------------------------------------------------#
#                               BLOCK  CONSTANTS                               #
# -----------------------------------------------------------------------------#

# FLUX conceptual blocks. We support BOTH:
# - model-style module paths: double_blocks.*, single_blocks.*, *_in, final_layer
# - LoRA-exporter-style keys: lora_transformer_(single_)?transformer_blocks_<n>_*
FLUX_BLOCK_NAME_MAPPING: Dict[str, List[str]] = {
    "Text Conditioning": ["txt_in."],
    "Timestep Embedding": ["time_in."],
    "Image Hint": ["img_in."],
    "Guidance Embedding": ["guidance_in."],
    "Vector Embedding": ["vector_in."],
    # Down path (model-style tokens)
    "Early Downsampling (Composition)": [f"double_blocks.{i}." for i in range(0, 4)],
    "Mid Downsampling (Subject & Concept)": [f"double_blocks.{i}." for i in range(4, 8)],
    "Late Downsampling (Refinement)": [f"double_blocks.{i}." for i in range(8, 10)],
    # Core stack (model-style tokens)
    "Core/Middle Block (Style Focus)": (
        [f"single_blocks.{i}." for i in range(0, 8)] +
        [f"double_blocks.{i}." for i in range(10, 19)]
    ),
    # Up path (model-style tokens)
    "Early Upsampling (Initial Style)": [f"double_blocks.{i}." for i in range(19, 23)] +
                                        [f"single_blocks.{i}." for i in range(8, 16)],
    "Mid Upsampling (Detail Generation)": [f"double_blocks.{i}." for i in range(23, 27)] +
                                        [f"single_blocks.{i}." for i in range(16, 32)],
    "Late Upsampling (Final Textures)": [f"double_blocks.{i}." for i in range(27, 29)] +
                                        [f"single_blocks.{i}." for i in range(32, 38)],
    # Output head
    "Final Output Layer (Latent Projection)": ["final_layer."],
    "Other Tensors": [],
}
ALL_FLUX_BLOCKS = list(FLUX_BLOCK_NAME_MAPPING.keys())

# SDXL blocks (coarse sliders)
SDXL_TEXT_ENCODER = "Text Encoder"
SDXL_INPUT_BLOCKS = "Input Blocks"
SDXL_MIDDLE_BLOCK = "Middle Block"
SDXL_OUTPUT_BLOCKS = "Output Blocks"
ALL_SDXL_BLOCKS = [
    SDXL_TEXT_ENCODER,
    SDXL_INPUT_BLOCKS,
    SDXL_MIDDLE_BLOCK,
    SDXL_OUTPUT_BLOCKS,
]

# -----------------------------------------------------------------------------#
#                             PRESET  DEFINITIONS                              #
# -----------------------------------------------------------------------------#

LORA_BLOCK_PRESETS = {
    "FLUX": {
        "Custom": {},
        "Full (Normal LoRA)": {
            "strength": 1.0,
            "block_weights": {name: 1.0 for name in ALL_FLUX_BLOCKS},
        },
        "Character": {
            "strength": 1.0,
            "block_weights": {
                "Text Conditioning": 1.0,
                "Early Downsampling (Composition)": 0.6,
                "Mid Downsampling (Subject & Concept)": 1.0,
                "Late Downsampling (Refinement)": 0.4,
                "Core/Middle Block (Style Focus)": 1.0,
                "Early Upsampling (Initial Style)": 0.1,
                "Mid Upsampling (Detail Generation)": 0.0,
                "Late Upsampling (Final Textures)": 0.0,
                "Final Output Layer (Latent Projection)": 0.0,
            },
        },
        "Style": {
            "strength": 1.0,
            "block_weights": {
                "Text Conditioning": 0.2,
                "Early Downsampling (Composition)": 0.1,
                "Mid Downsampling (Subject & Concept)": 0.0,
                "Late Downsampling (Refinement)": 0.2,
                "Core/Middle Block (Style Focus)": 0.5,
                "Early Upsampling (Initial Style)": 1.0,
                "Mid Upsampling (Detail Generation)": 1.0,
                "Late Upsampling (Final Textures)": 1.0,
                "Final Output Layer (Latent Projection)": 1.0,
            },
        },
        "Concept": {
            "strength": 1.0,
            "block_weights": {
                "Text Conditioning": 1.0,
                "Early Downsampling (Composition)": 1.0,
                "Mid Downsampling (Subject & Concept)": 0.9,
                "Late Downsampling (Refinement)": 0.6,
                "Core/Middle Block (Style Focus)": 0.7,
                "Early Upsampling (Initial Style)": 0.5,
                "Mid Upsampling (Detail Generation)": 0.3,
                "Late Upsampling (Final Textures)": 0.1,
                "Final Output Layer (Latent Projection)": 0.0,
            },
        },
        "Fix Hands/Anatomy": {
            "strength": 0.4,
            "block_weights": {
                "Text Conditioning": 0.2,
                "Early Downsampling (Composition)": 1.0,
                "Mid Downsampling (Subject & Concept)": 0.3,
                "Late Downsampling (Refinement)": 0.0,
                "Core/Middle Block (Style Focus)": 0.0,
                "Early Upsampling (Initial Style)": 0.0,
                "Mid Upsampling (Detail Generation)": 0.0,
                "Late Upsampling (Final Textures)": 0.0,
                "Final Output Layer (Latent Projection)": 0.0,
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
            },
        },
        "Style": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.0,
                SDXL_INPUT_BLOCKS: 0.2,
                SDXL_MIDDLE_BLOCK: 0.5,
                SDXL_OUTPUT_BLOCKS: 1.0,
            },
        },
        "Concept": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 1.0,
                SDXL_INPUT_BLOCKS: 0.8,
                SDXL_MIDDLE_BLOCK: 0.7,
                SDXL_OUTPUT_BLOCKS: 0.5,
            },
        },
        "Fix Hands/Anatomy": {
            "strength": 0.4,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.2,
                SDXL_INPUT_BLOCKS: 1.0,
                SDXL_MIDDLE_BLOCK: 0.4,
                SDXL_OUTPUT_BLOCKS: 0.0,
            },
        },
    },
}

# -----------------------------------------------------------------------------#
#                               HELPER FUNCTIONS                               #
# -----------------------------------------------------------------------------#

def _build_key_map(model, clip) -> Dict[str, Any]:
    """Build key_map compatibly across ComfyUI versions."""
    key_map: Dict[str, Any] = {}
    if hasattr(comfy.lora, "model_lora_keys_unet"):
        comfy.lora.model_lora_keys_unet(model.model, key_map)
    if hasattr(comfy.lora, "model_lora_keys_clip"):
        comfy.lora.model_lora_keys_clip(clip.cond_stage_model, key_map)
    return key_map

def _invert_key_map(key_map: Dict[str, Any]) -> Dict[Any, str]:
    """Invert key_map from raw_name -> (module, ...) to (module) -> raw_name."""
    inv: Dict[Any, str] = {}
    for raw, info in key_map.items():
        inv[info[0]] = raw  # info[0] is the nn.Module
    return inv

def _example_raws_from_patches(patches: Dict[Tuple[Any, ...], Any],
                               inv_key_map: Dict[Any, str],
                               limit: int = 3) -> List[str]:
    out: List[str] = []
    for key_tuple in patches.keys():
        mod = key_tuple[0]
        raw = inv_key_map.get(mod, "")
        if raw:
            out.append(raw.split(".", 1)[-1])
        if len(out) >= limit:
            break
    return out

# ----------------------------- SDXL grouping -------------------------------- #

def _split_sdxl_patches_by_block(loaded_patches: Dict[Any, Any],
                                 inv_key_map: Dict[Any, str]):
    """Split patches into SDXL input/middle/output/clip groups based on raw weight names."""
    p_in, p_mid, p_out, p_clip, p_other = {}, {}, {}, {}, {}
    for key_tuple, patch in loaded_patches.items():
        raw_key = inv_key_map.get(key_tuple[0], "")
        rk = raw_key  # permissive contains-checks handle exporter differences
        if "input_blocks" in rk:
            p_in[key_tuple] = patch
        elif "middle_block" in rk:
            p_mid[key_tuple] = patch
        elif "output_blocks" in rk:
            p_out[key_tuple] = patch
        elif ("text_encoder" in rk) or ("lora_te" in rk) or ("clip" in rk and "proj" in rk):
            p_clip[key_tuple] = patch
        else:
            p_other[key_tuple] = patch
    return p_in, p_mid, p_out, p_clip, p_other

def _log_sdxl_counts(logger: logging.Logger,
                     p_in: Dict[Any, Any], p_mid: Dict[Any, Any],
                     p_out: Dict[Any, Any], p_clip: Dict[Any, Any],
                     p_other: Dict[Any, Any],
                     te_strength: float, in_strength: float,
                     mid_strength: float, out_strength: float,
                     inv_key_map: Dict[Any, str]):
    cnt_te = len(p_clip)
    cnt_in = len(p_in)
    cnt_mid = len(p_mid)
    cnt_out = len(p_out)
    cnt_other = len(p_other)
    logger.info(
        f"[SDXL] Patch counts — TE:{cnt_te} IN:{cnt_in} MID:{cnt_mid} OUT:{cnt_out} OTHER:{cnt_other} "
        f"| strengths TE={te_strength:.2f} IN={in_strength:.2f} MID={mid_strength:.2f} OUT={out_strength:.2f}"
    )

    def _why_zero(name: str, cnt: int, strength: float, sample_from: Dict[Any, Any]):
        if cnt > 0:
            return
        if strength == 0.0:
            logger.info(f"[SDXL] {name}: 0 patches applied because strength is 0.00.")
        else:
            examples = _example_raws_from_patches(sample_from, inv_key_map, 3)
            logger.info(
                f"[SDXL] {name}: 0 patches found in this LoRA (strength {strength:.2f} > 0). "
                f"Likely the LoRA was trained without deltas for this block.\n"
                f"Counts — TE:{cnt_te} IN:{cnt_in} MID:{cnt_mid} OUT:{cnt_out} OTHER:{cnt_other}. "
                f"Example raw tails present: {examples}"
            )

    sample_source = {}
    for d in (p_in, p_mid, p_out, p_clip, p_other):
        if len(d) > len(sample_source):
            sample_source = d

    _why_zero("Input Blocks", cnt_in, in_strength, sample_source)
    _why_zero("Middle Block", cnt_mid, mid_strength, sample_source)
    _why_zero("Output Blocks", cnt_out, out_strength, sample_source)

# ----------------------------- FLUX grouping -------------------------------- #

# Normalize keys to underscore style, then regex them.
def _normalize_key(key: str) -> str:
    """Lowercase and replace separators with underscores for uniform matching."""
    k = key.replace("/", "_").replace(".", "_").replace(":", "_").lower()
    k = re.sub(r"__+", "_", k)
    return k

# Heuristic splitter for Flux exporter keys (plus model-style).
# Matches variations like:
#   transformer_blocks_<IDX>_...
#   lora_transformer_transformer_blocks_<IDX>_...
#   single_transformer_blocks_<IDX>_...
#   lora_transformer_single_transformer_blocks_<IDX>_...
# And also model-style paths like:
#   double_blocks.<i>.
#   single_blocks.<i>.
_RE_TX_BLOCK_US = re.compile(r"(?:^|_)(?:lora_transformer_)?transformer_blocks_(\d+)(?:_|$)")
_RE_SINGLE_TX_BLOCK_US = re.compile(r"(?:^|_)(?:lora_transformer_)?single_transformer_blocks_(\d+)(?:_|$)")
_RE_DOUBLE_BLOCKS_US = re.compile(r"(?:^|_)(?:lora_transformer_)?double_blocks_(\d+)(?:_|$)")
_RE_SINGLE_BLOCKS_US = re.compile(r"(?:^|_)(?:lora_transformer_)?single_blocks_(\d+)(?:_|$)")

def _flux_concept_from_raw(raw_key: str) -> str:
    """
    Map a raw module key to a conceptual FLUX block. Supports both:
      - model-style paths (double_blocks.*, single_blocks.*, *_in, final_layer)
      - LoRA-exporter-style names (transformer_blocks_#, single_transformer_blocks_#)
    """
    rk_norm = _normalize_key(raw_key)

    # Inlet/head tokens (string contains checks on normalized)
    if "_txt_in" in rk_norm or "time_text_embed" in rk_norm: return "Text Conditioning"
    if "_time_in" in rk_norm: return "Timestep Embedding"
    if "_img_in" in rk_norm: return "Image Hint"
    if "_guidance_in" in rk_norm: return "Guidance Embedding"
    if "_vector_in" in rk_norm: return "Vector Embedding"
    if "_final_layer" in rk_norm: return "Final Output Layer (Latent Projection)"

    # LoRA-exporter: single_transformer_blocks_*
    m_single_tx = _RE_SINGLE_TX_BLOCK_US.search(rk_norm)
    if m_single_tx:
        idx = int(m_single_tx.group(1))
        if 0 <= idx < 8: return "Core/Middle Block (Style Focus)"
        if 8 <= idx < 16: return "Early Upsampling (Initial Style)"
        if 16 <= idx < 32: return "Mid Upsampling (Detail Generation)"
        if 32 <= idx < 38: return "Late Upsampling (Final Textures)"

    # Model-style single_blocks
    m_sb = _RE_SINGLE_BLOCKS_US.search(rk_norm)
    if m_sb:
        idx = int(m_sb.group(1))
        if 0 <= idx < 8: return "Core/Middle Block (Style Focus)"
        if 8 <= idx < 16: return "Early Upsampling (Initial Style)"
        if 16 <= idx < 32: return "Mid Upsampling (Detail Generation)"
        if 32 <= idx < 38: return "Late Upsampling (Final Textures)"

    # Model-style double_blocks indices
    m_db = _RE_DOUBLE_BLOCKS_US.search(rk_norm)
    if m_db:
        idx = int(m_db.group(1))
        if 0 <= idx < 4:  return "Early Downsampling (Composition)"
        if 4 <= idx < 8:  return "Mid Downsampling (Subject & Concept)"
        if 8 <= idx < 10:  return "Late Downsampling (Refinement)"
        if 10 <= idx < 19: return "Core/Middle Block (Style Focus)"
        if 19 <= idx < 23: return "Early Upsampling (Initial Style)"
        if 23 <= idx < 27: return "Mid Upsampling (Detail Generation)"
        if 27 <= idx < 29: return "Late Upsampling (Final Textures)"

    # LoRA-exporter: transformer_blocks_IDX => map ranges (double path)
    m_tx = _RE_TX_BLOCK_US.search(rk_norm)
    if m_tx:
        idx = int(m_tx.group(1))
        if 0 <= idx < 4:  return "Early Downsampling (Composition)"
        if 4 <= idx < 8:  return "Mid Downsampling (Subject & Concept)"
        if 8 <= idx < 10:  return "Late Downsampling (Refinement)"
        if 10 <= idx < 19: return "Core/Middle Block (Style Focus)"
        if 19 <= idx < 23: return "Early Upsampling (Initial Style)"
        if 23 <= idx < 27: return "Mid Upsampling (Detail Generation)"
        if 27 <= idx < 29: return "Late Upsampling (Final Textures)"

    return "Other Tensors"

def _group_flux_patches(loaded_patches: Dict[Any, Any],
                        inv_key_map: Dict[Any, str]) -> Dict[str, Dict[Any, Any]]:
    groups = {name: {} for name in ALL_FLUX_BLOCKS}
    for key_tuple, patch in loaded_patches.items():
        raw_key = inv_key_map.get(key_tuple[0], "")
        concept = _flux_concept_from_raw(raw_key) if raw_key else "Other Tensors"
        if concept in groups:
            groups[concept][key_tuple] = patch
        else:
            groups["Other Tensors"][key_tuple] = patch
    return groups

def _log_flux_counts(logger: logging.Logger,
                     groups: Dict[str, Dict[Any, Any]],
                     strengths: Dict[str, float],
                     inv_key_map: Dict[Any, str]):
    logger.info("[FLUX] Patch counts per block:")
    total = 0
    for name in ALL_FLUX_BLOCKS:
        cnt = len(groups.get(name, {}))
        total += cnt
        logger.info(f"    - {name}: {cnt} (strength {strengths.get(name, 0.0):.2f})")
    logger.info(f"[FLUX] Total matched patches: {total}")

    # Explain zeros
    for name in ALL_FLUX_BLOCKS:
        patches = groups.get(name, {})
        strength = strengths.get(name, 0.0)
        if len(patches) > 0:
            continue
        if strength == 0.0:
            logger.info(f"[FLUX] {name}: 0 patches applied because strength is 0.00.")
        else:
            # Use the largest non-empty group as a sample source
            non_empty = [(k, v) for k, v in groups.items() if len(v) > 0]
            sample = non_empty[0][1] if non_empty else {}
            examples = _example_raws_from_patches(sample, inv_key_map, 3)
            logger.info(
                f"[FLUX] {name}: 0 patches in this LoRA (strength {strength:.2f} > 0). "
                f"LoRA likely doesn't include weights for this block. "
                f"Sample mapped raw tails: {examples}"
            )
            
# -----------------------------------------------------------------------------#
#                               FLUX  LOADER                                   #
# -----------------------------------------------------------------------------#

class BobsLoraLoaderFlux:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            "preset": (list(LORA_BLOCK_PRESETS["FLUX"].keys()),),
        }
        for blk in ALL_FLUX_BLOCKS:
            req[blk] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05})
        return {"required": req}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_lora"
    CATEGORY = "Bobs_Nodes"

    def apply_lora(self, model, clip, lora_name, strength, preset, **kwargs):

        if lora_name == "None" or strength == 0.0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            self.logger.error(f"[FLUX] LoRA file not found: {lora_name}")
            return model, clip

        self.logger.info(f"[FLUX] Loading LoRA: {lora_name}")
        if os.path.splitext(lora_path)[1].lower() == ".safetensors":
            lora_sd = safe_load_file(lora_path, device="cpu")
        else:
            lora_sd = torch.load(lora_path, map_location="cpu")

        # -------- build per-block final strengths ----------
        block_strength: Dict[str, float] = {}
        if preset == "Custom":
            for blk in ALL_FLUX_BLOCKS:
                block_strength[blk] = float(kwargs.get(blk, 1.0)) * float(strength)
        else:
            cfg = LORA_BLOCK_PRESETS["FLUX"][preset]
            base = float(strength) * float(cfg.get("strength", 1.0))
            weights = cfg.get("block_weights", {})
            for blk in ALL_FLUX_BLOCKS:
                block_strength[blk] = float(weights.get(blk, 1.0)) * base

        # -------- build key map & load patches -------------
        key_map: Dict[str, Any] = _build_key_map(model, clip)
        if not key_map:
            self.logger.warning("[FLUX] Empty key_map; LoRA will not be applied.")
            return model, clip

        inv_key_map = _invert_key_map(key_map)
        all_patches = comfy.lora.load_lora(lora_sd, key_map)

        if not all_patches:
            self.logger.warning("[FLUX] No matching keys in LoRA checkpoint.")
            return model, clip

        grouped_patches = _group_flux_patches(all_patches, inv_key_map)

        # -------- logging: counts & explanations -----------
        _log_flux_counts(self.logger, grouped_patches, block_strength, inv_key_map)

        # -------- clone & attach patches group by group ----
        out_model = model.clone()
        out_clip = clip.clone()

        for concept, patches_in_group in grouped_patches.items():
            s = block_strength.get(concept, 0.0)
            if s == 0.0 or not patches_in_group:
                continue
            # Apply to both model and clip; non-matching targets are no-ops
            out_model.add_patches(patches_in_group, s)
            out_clip.add_patches(patches_in_group, s)

        return out_model, out_clip


# -----------------------------------------------------------------------------#
#                               SDXL  LOADER                                   #
# -----------------------------------------------------------------------------#

class BobsLoraLoaderSdxl:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
            "preset": (list(LORA_BLOCK_PRESETS["SDXL"].keys()),),
        }
        for blk in ALL_SDXL_BLOCKS:
            req[blk] = ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05})
        return {"required": req}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_lora"
    CATEGORY = "Bobs_Nodes"

    def apply_lora(self, model, clip, lora_name, strength, preset, **kwargs):

        if lora_name == "None" or strength == 0.0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            self.logger.error(f"[SDXL] LoRA file not found: {lora_name}")
            return model, clip

        self.logger.info(f"[SDXL] Loading LoRA: {lora_name}")
        if os.path.splitext(lora_path)[1].lower() == ".safetensors":
            lora_sd = safe_load_file(lora_path, device="cpu")
        else:
            lora_sd = torch.load(lora_path, map_location="cpu")

        # --- Build block strengths ---
        if preset == "Custom":
            te_strength = float(kwargs.get(SDXL_TEXT_ENCODER, 1.0)) * float(strength)
            in_strength = float(kwargs.get(SDXL_INPUT_BLOCKS, 1.0)) * float(strength)
            mid_strength = float(kwargs.get(SDXL_MIDDLE_BLOCK, 1.0)) * float(strength)
            out_strength = float(kwargs.get(SDXL_OUTPUT_BLOCKS, 1.0)) * float(strength)
        else:
            cfg = LORA_BLOCK_PRESETS["SDXL"][preset]
            base = float(strength) * float(cfg.get("strength", 1.0))
            weights = cfg.get("block_weights", {})
            te_strength  = float(weights.get(SDXL_TEXT_ENCODER, 1.0)) * base
            in_strength  = float(weights.get(SDXL_INPUT_BLOCKS, 1.0)) * base
            mid_strength = float(weights.get(SDXL_MIDDLE_BLOCK, 1.0)) * base
            out_strength = float(weights.get(SDXL_OUTPUT_BLOCKS, 1.0)) * base

        # --- Build key map for model and clip ---
        key_map: Dict[str, Any] = _build_key_map(model, clip)
        if not key_map:
            self.logger.warning("[SDXL] Empty key_map; LoRA will not be applied.")
            return model, clip

        inv_key_map = _invert_key_map(key_map)
        all_patches = comfy.lora.load_lora(lora_sd, key_map)

        if not all_patches:
            self.logger.warning("[SDXL] No matching keys in LoRA checkpoint.")
            return model, clip

        p_in, p_mid, p_out, p_clip, p_other = _split_sdxl_patches_by_block(all_patches, inv_key_map)

        # --- Log counts and zero reasons ---
        _log_sdxl_counts(self.logger, p_in, p_mid, p_out, p_clip, p_other,
                         te_strength, in_strength, mid_strength, out_strength,
                         inv_key_map)

        # --- Apply ---
        out_model = model.clone()
        out_clip = clip.clone()

        if p_in and in_strength != 0.0:
            out_model.add_patches(p_in, in_strength)
        if p_mid and mid_strength != 0.0:
            out_model.add_patches(p_mid, mid_strength)
        if p_out and out_strength != 0.0:
            out_model.add_patches(p_out, out_strength)
        if p_clip and te_strength != 0.0:
            out_clip.add_patches(p_clip, te_strength)

        return (out_model, out_clip)


# -----------------------------------------------------------------------------#
#                             COMFYUI REGISTRATION                             #
# -----------------------------------------------------------------------------#

NODE_CLASS_MAPPINGS = {
    "BobsLoraLoaderFlux": BobsLoraLoaderFlux,
    "BobsLoraLoaderSdxl": BobsLoraLoaderSdxl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BobsLoraLoaderFlux": "Bobs LoRA Loader (FLUX)",
    "BobsLoraLoaderSdxl": "Bobs LoRA Loader (SDXL)",
}
