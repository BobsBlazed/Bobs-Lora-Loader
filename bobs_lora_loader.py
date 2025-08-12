import logging
from typing import Dict, Any, List
import os

import comfy.utils
import comfy.lora
import folder_paths
import torch
from safetensors.torch import load_file as safe_load_file


# -----------------------------------------------------------------------------#
#                               BLOCK  CONSTANTS                               #
# -----------------------------------------------------------------------------#


FLUX_BLOCK_NAME_MAPPING = {
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

RAW_TO_CONCEPT_MAPPING = {
    raw: concept
    for concept, raws in FLUX_BLOCK_NAME_MAPPING.items()
    for raw in raws
}

ALL_FLUX_BLOCKS = list(FLUX_BLOCK_NAME_MAPPING.keys())

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
                SDXL_OUTPUT_BLOCKS:0.2,
            },
        },
        "Style": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.0,
                SDXL_INPUT_BLOCKS: 0.2,
                SDXL_MIDDLE_BLOCK: 0.5,
                SDXL_OUTPUT_BLOCKS:1.0,
            },
        },
        "Concept": {
            "strength": 1.0,
            "block_weights": {
                SDXL_TEXT_ENCODER: 1.0,
                SDXL_INPUT_BLOCKS: 0.8,
                SDXL_MIDDLE_BLOCK: 0.7,
                SDXL_OUTPUT_BLOCKS:0.5,
            },
        },
        "Fix Hands/Anatomy": {
            "strength": 0.4,
            "block_weights": {
                SDXL_TEXT_ENCODER: 0.2,
                SDXL_INPUT_BLOCKS: 1.0,
                SDXL_MIDDLE_BLOCK: 0.4,
                SDXL_OUTPUT_BLOCKS:0.0,
            },
        },
    },
}


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
        if os.path.splitext(lora_path)[1] == ".safetensors":
            lora_sd = safe_load_file(lora_path, device="cpu")
        else:
            lora_sd = torch.load(lora_path, map_location="cpu")

        # -------- build per-block final strengths ----------
        block_strength: Dict[str, float] = {}
        if preset == "Custom":
            for blk in ALL_FLUX_BLOCKS:
                block_strength[blk] = kwargs.get(blk, 1.0) * strength
        else:
            cfg = LORA_BLOCK_PRESETS["FLUX"][preset]
            base = strength * cfg.get("strength", 1.0)
            weights = cfg.get("block_weights", {})
            for blk in ALL_FLUX_BLOCKS:
                block_strength[blk] = weights.get(blk, 1.0) * base

        # -------- build key map  ---------------------------
        key_map: Dict[str, Any] = {}
        key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
        key_map.update(comfy.lora.model_lora_keys_clip(clip.cond_stage_model, {}))
        mod_to_raw = {info[0]: raw for raw, info in key_map.items()}

        # -------- load patches -----------------------------
        all_patches = comfy.lora.load_lora(lora_sd, key_map)
        if not all_patches:
            self.logger.warning("[FLUX] No matching keys in LoRA checkpoint.")
            return model, clip

        # -------- Group patches by concept -----------------
        grouped_patches: Dict[str, Dict] = {name: {} for name in ALL_FLUX_BLOCKS}

        for key_tuple, patch in all_patches.items():
            module = key_tuple[0]
            raw_key = mod_to_raw.get(module, "")

            # ------ Classify the patch into a concept group ------
            if not raw_key.startswith("diffusion_model."):
                concept = "Text Conditioning"
            else:
                tail = raw_key[len("diffusion_model.") :]
                found_concept = "Other Tensors"
                for prefix, cname in RAW_TO_CONCEPT_MAPPING.items():
                    if tail.startswith(prefix):
                        found_concept = cname
                        break
                concept = found_concept

            if concept in grouped_patches:
                grouped_patches[concept][key_tuple] = patch

        # -------- clone & attach patches group by group ------
        out_model = model.clone()
        out_clip = clip.clone()

        for concept, patches_in_group in grouped_patches.items():
            strength_for_group = block_strength.get(concept, 0.0)

            if strength_for_group == 0.0 or not patches_in_group:
                continue
            
            out_model.add_patches(patches_in_group, strength_for_group)
            out_clip.add_patches(patches_in_group, strength_for_group)

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
        if os.path.splitext(lora_path)[1] == ".safetensors":
            lora_sd = safe_load_file(lora_path, device="cpu")
        else:
            lora_sd = torch.load(lora_path, map_location="cpu")
        
        # Build key_map compatibly across ComfyUI versions
        key_map = {}
        try:
            # Newer ComfyUI exposes model_lora_keys(model, clip)
            key_map, _ = comfy.lora.model_lora_keys(model, clip)
        except AttributeError:
            # Older ComfyUI: compose map from UNet and CLIP
            unet = getattr(model, "model", model)
            clipm = getattr(clip, "cond_stage_model", clip)
            if hasattr(comfy.lora, "model_lora_keys_unet"):
                key_map = comfy.lora.model_lora_keys_unet(unet, key_map)
            if hasattr(comfy.lora, "model_lora_keys_clip"):
                key_map.update(comfy.lora.model_lora_keys_clip(clipm, {}))

        block_strength: Dict[str, float] = {}
        if preset == "Custom":
            for blk in ALL_SDXL_BLOCKS:
                block_strength[blk] = kwargs.get(blk, 1.0) * strength
        else:
            cfg = LORA_BLOCK_PRESETS["SDXL"][preset]
            base = strength * cfg.get("strength", 1.0)
            weights = cfg.get("block_weights", {})
            for blk in ALL_SDXL_BLOCKS:
                block_strength[blk] = weights.get(blk, 1.0) * base

        # Apply with block weights if available; otherwise add patches directly
        if hasattr(comfy.lora, "load_lora_for_models_with_block_weights"):
            model_new, clip_new = comfy.lora.load_lora_for_models_with_block_weights(
                model, clip, comfy.lora.load_lora(lora_sd, key_map), 1.0, 1.0, block_strength
            )
            return (model_new, clip_new)
        else:
            # Very old fallback
            patches = comfy.lora.load_lora(lora_sd, key_map)
            out_model = model.clone()
            out_clip = clip.clone()
            # Approximate by applying a single global strength using the middle block weight as a proxy
            approx_strength = block_strength.get(SDXL_MIDDLE_BLOCK, 1.0)
            out_model.add_patches(patches, approx_strength)
            out_clip.add_patches(patches, approx_strength)
            return out_model, out_clip


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
