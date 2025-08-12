# bobs_lora_loader.py
# Bobs LoRA Loader (SDXL + FLUX) — ComfyUI 0.3.x compatible
# - SDXL node supports block-weighted LoRA apply (input/middle/output) + Text Encoder strength
# - FLUX node applies LoRA with fine-grained block controls mapped to Flux module name patterns
#
# Uses only public ComfyUI APIs:
#   - comfy.lora.model_lora_keys / model_lora_keys_unet / model_lora_keys_clip
#   - comfy.lora.load_lora (returns patch dict)
#   - ModelPatcher.add_patches(...) and .clone()
#
# License: MIT

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple

import os
import torch
import comfy.utils
import comfy.lora
import folder_paths
from safetensors.torch import load_file as safe_load_file

LOGGER = logging.getLogger("BobsLoRALoader")
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Flux block naming maps (for fine-grained control)
# -----------------------------------------------------------------------------

FLUX_BLOCK_NAME_MAPPING: Dict[str, list] = {
    "Text Conditioning": ["txt_in."],
    "Timestep Embedding": ["time_in."],
    "Image Hint": ["img_in."],
    "Early Downsampling (Composition)": [f"double_blocks.{i}." for i in range(4)],
    "Mid Downsampling (Subject & Concept)": [f"double_blocks.{i}." for i in range(4, 8)],
    "Late Downsampling (Refinement)": [f"double_blocks.{i}." for i in range(8, 10)],
    "Core/Middle Block (Style Focus)": (
        [f"single_blocks.{i}." for i in range(38)]
        + [f"double_blocks.{i}." for i in range(10, 19)]
    ),
    "Early Upsampling (Initial Style)": [f"double_blocks.{i}." for i in range(19, 23)],
    "Mid Upsampling (Detail Generation)": [f"double_blocks.{i}." for i in range(23, 27)],
    "Late Upsampling (Final Textures)": [f"double_blocks.{i}." for i in range(27, 29)],
    "Final Output Layer (Latent Projection)": ["final_layer."],
    "Other Tensors": [],
}

RAW_TO_CONCEPT_MAPPING: Dict[str, str] = {
    raw: concept
    for concept, raws in FLUX_BLOCK_NAME_MAPPING.items()
    for raw in raws
}

ALL_FLUX_BLOCKS = list(FLUX_BLOCK_NAME_MAPPING.keys())

# -----------------------------------------------------------------------------
# SDXL constants and presets
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _build_key_map(model, clip) -> Dict[str, Any]:
    """Build key_map compatibly across ComfyUI versions."""
    # Try unified helper
    if hasattr(comfy.lora, "model_lora_keys"):
        try:
            key_map, _ = comfy.lora.model_lora_keys(model, clip)
            return key_map
        except Exception as e:
            LOGGER.debug(f"Unified model_lora_keys failed, falling back: {e}")

    # Fallback: per-module helpers
    key_map = {}
    try:
        if model is not None and hasattr(comfy.lora, "model_lora_keys_unet"):
            unet = getattr(model, "model", model)
            comfy.lora.model_lora_keys_unet(unet, key_map)
        if clip is not None and hasattr(comfy.lora, "model_lora_keys_clip"):
            clipm = getattr(clip, "cond_stage_model", clip)
            comfy.lora.model_lora_keys_clip(clipm, key_map)
    except Exception as e:
        LOGGER.warning(f"Failed to build LoRA key_map: {e}")
    return key_map


def _invert_key_map(key_map: Dict[str, Any]) -> Dict[Any, str]:
    """Invert key_map from raw_name -> key_tuple to key_tuple -> raw_name."""
    inv = {}
    for raw, key in key_map.items():
        inv[key] = raw
    return inv


def _split_sdxl_unet_by_block(loaded_patches: Dict[Any, Any], inv_key_map: Dict[Any, str]):
    """Split patches into SDXL input/middle/output/other groups based on raw weight names."""
    p_in, p_mid, p_out, p_other, p_clip = {}, {}, {}, {}, {}
    for key_tuple, patch in loaded_patches.items():
        raw = inv_key_map.get(key_tuple, "")
        if isinstance(raw, str) and raw.startswith("diffusion_model."):
            tail = raw[len("diffusion_model."):]
            if tail.startswith("input_blocks."):
                p_in[key_tuple] = patch
            elif tail.startswith("middle_block."):
                p_mid[key_tuple] = patch
            elif tail.startswith("output_blocks."):
                p_out[key_tuple] = patch
            else:
                p_other[key_tuple] = patch
        else:
            # Not diffusion_model.* -> likely CLIP/text enc
            p_clip[key_tuple] = patch
    return p_in, p_mid, p_out, p_other, p_clip


def _group_flux_patches(loaded_patches: Dict[Any, Any], inv_key_map: Dict[Any, str]):
    """Group Flux patches by conceptual block using name prefixes."""
    groups = {name: {} for name in ALL_FLUX_BLOCKS}
    for key_tuple, patch in loaded_patches.items():
        raw = inv_key_map.get(key_tuple, "")
        if not isinstance(raw, str):
            continue

        if not raw.startswith("diffusion_model."):
            concept = "Text Conditioning"
        else:
            tail = raw[len("diffusion_model."):]
            concept = "Other Tensors"
            for prefix, cname in RAW_TO_CONCEPT_MAPPING.items():
                if tail.startswith(prefix):
                    concept = cname
                    break
        if concept in groups:
            groups[concept][key_tuple] = patch
    return groups


# -----------------------------------------------------------------------------
# FLUX Loader
# -----------------------------------------------------------------------------

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
    CATEGORY = "Bobs/Loaders"

    def apply_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        if lora_name == "None" or strength == 0.0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            self.logger.error(f"[FLUX] LoRA file not found: {lora_name}")
            return model, clip

        self.logger.info(f"[FLUX] Loading LoRA: {lora_name}")
        if os.path.splitext(lora_path)[1] == ".safetensors":
            lora_sd = safe_load_file(lora_path, device="cpu")
        else:
            lora_sd = torch.load(lora_path, map_location="cpu")

        # preset/custom weights
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

        # map + load patches
        key_map = _build_key_map(model, clip)
        if not key_map:
            self.logger.warning("[FLUX] Empty key_map; LoRA will not be applied.")
            return model, clip

        loaded_patches = comfy.lora.load_lora(lora_sd, key_map)
        if not loaded_patches:
            self.logger.warning("[FLUX] No matching keys in LoRA checkpoint.")
            return model, clip

        inv_key_map = _invert_key_map(key_map)
        grouped = _group_flux_patches(loaded_patches, inv_key_map)

        out_model = model.clone()
        out_clip = clip.clone()

        # Apply per group with its strength
        for concept, patches_in_group in grouped.items():
            s = float(block_strength.get(concept, 0.0))
            if s == 0.0 or not patches_in_group:
                continue
            try:
                out_model.add_patches(patches_in_group, strength_patch=s, strength_model=1.0)
            except Exception as e:
                self.logger.debug(f"[FLUX] model add_patches failed for '{concept}': {e}")
            try:
                out_clip.add_patches(patches_in_group, strength_patch=s, strength_model=1.0)
            except Exception as e:
                self.logger.debug(f"[FLUX] clip add_patches failed for '{concept}': {e}")

        return out_model, out_clip


# -----------------------------------------------------------------------------
# SDXL Loader
# -----------------------------------------------------------------------------

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
    CATEGORY = "Bobs/Loaders"

    def apply_lora(self, model, clip, lora_name, strength, preset, **kwargs):
        if lora_name == "None" or strength == 0.0:
            return model, clip

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            self.logger.error(f"[SDXL] LoRA file not found: {lora_name}")
            return model, clip

        self.logger.info(f"[SDXL] Loading LoRA: {lora_name}")
        if os.path.splitext(lora_path)[1] == ".safetensors":
            lora_sd = safe_load_file(lora_path, device="cpu")
        else:
            lora_sd = torch.load(lora_path, map_location="cpu")

        # Strengths
        te_slider = float(kwargs.get(SDXL_TEXT_ENCODER, 1.0))
        in_slider = float(kwargs.get(SDXL_INPUT_BLOCKS, 1.0))
        mid_slider = float(kwargs.get(SDXL_MIDDLE_BLOCK, 1.0))
        out_slider = float(kwargs.get(SDXL_OUTPUT_BLOCKS, 1.0))

        if preset != "Custom":
            cfg = LORA_BLOCK_PRESETS["SDXL"][preset]
            base = float(strength) * float(cfg.get("strength", 1.0))
            weights = cfg.get("block_weights", {})
            te_strength = float(weights.get(SDXL_TEXT_ENCODER, 1.0)) * base
            in_strength = float(weights.get(SDXL_INPUT_BLOCKS, 1.0)) * base
            mid_strength = float(weights.get(SDXL_MIDDLE_BLOCK, 1.0)) * base
            out_strength = float(weights.get(SDXL_OUTPUT_BLOCKS, 1.0)) * base
        else:
            base = float(strength)
            te_strength = te_slider * base
            in_strength = in_slider * base
            mid_strength = mid_slider * base
            out_strength = out_slider * base

        # Map + load patches
        key_map = _build_key_map(model, clip)
        if not key_map:
            self.logger.warning("[SDXL] Empty key_map; LoRA will not be applied.")
            return model, clip

        loaded_patches = comfy.lora.load_lora(lora_sd, key_map)
        if not loaded_patches:
            self.logger.warning("[SDXL] No matching keys in LoRA checkpoint.")
            return model, clip

        inv_key_map = _invert_key_map(key_map)

        p_in, p_mid, p_out, p_other, p_clip = _split_sdxl_unet_by_block(loaded_patches, inv_key_map)
        try:
            self.logger.info(f"[SDXL] LoRA keys: input={len(p_in)} middle={len(p_mid)} output={len(p_out)} other={len(p_other)} clip={len(p_clip)}")
        except Exception:
            pass

        out_model = model.clone()
        out_clip = clip.clone()

        # Apply UNet patches
        if p_in and in_strength != 0.0:
            out_model.add_patches(p_in, strength_patch=in_strength, strength_model=1.0)
        if p_mid and mid_strength != 0.0:
            out_model.add_patches(p_mid, strength_patch=mid_strength, strength_model=1.0)
        if p_out and out_strength != 0.0:
            out_model.add_patches(p_out, strength_patch=out_strength, strength_model=1.0)
        if p_other:
            # time_embed etc. get base strength
            out_model.add_patches(p_other, strength_patch=float(strength), strength_model=1.0)

        # Apply CLIP patches (Text Encoder)
        if p_clip and te_strength != 0.0:
            out_clip.add_patches(p_clip, strength_patch=te_strength, strength_model=1.0)

        return out_model, out_clip


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "BobsLoraLoaderFlux": BobsLoraLoaderFlux,
    "BobsLoraLoaderSdxl": BobsLoraLoaderSdxl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BobsLoraLoaderFlux": "Bobs LoRA Loader (FLUX)",
    "BobsLoraLoaderSdxl": "Bobs LoRA Loader (SDXL)",
}

def _announce():
    try:
        print("✨ Bobs LoRA Loader nodes loaded! ✨")
    except Exception:
        pass

_announce()
