"""
Bobs LoRA Loader — block-weighted LoRA loading for ComfyUI (FLUX + SDXL).

How this works
--------------
ComfyUI's ``comfy.lora`` builds a ``key_map`` that translates every LoRA
exporter dialect (kohya ``lora_unet_*``, OneTrainer ``lora_transformer_*``,
diffusers ``transformer.*``, lycoris, DiffSynth, ...) into the *canonical model
state-dict key* the patch targets. ``comfy.lora.load_lora`` then returns a patch
dict keyed by those canonical names — either a plain string, or a
``(key, offset)`` tuple when several LoRA tensors are packed into one fused
weight (FLUX ``qkv``/``linear1``).

We therefore classify patches by their canonical target key rather than by the
raw name inside the LoRA file. That is dialect-proof: if ComfyUI can load the
LoRA at all, this node can bucket it.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import comfy.lora
import comfy.utils
import folder_paths

from .bobs_blocks import (
    ALL_FLUX_BLOCKS,
    ALL_SDXL_BLOCKS,
    BLOCK_TOOLTIPS,
    FLUX_DEFAULT_DEPTH,
    FLUX_DEFAULT_DEPTH_SINGLE,
    LORA_BLOCK_PRESETS,
    TEXT_ENCODER_BLOCK,
    classify_flux_key,
    classify_sdxl_key,
    flux_block_ranges,
    resolve_block_strengths,
)

try:  # Added to ComfyUI in 2025; handles BFL-control / Wan-Fun / USO LoRAs.
    import comfy.lora_convert as _lora_convert
except ImportError:  # pragma: no cover - older ComfyUI
    _lora_convert = None

logger = logging.getLogger("BobsLoraLoader")

MAX_STRENGTH = 5.0
MAX_BLOCK_WEIGHT = 2.0


# -----------------------------------------------------------------------------#
#                               HELPER FUNCTIONS                               #
# -----------------------------------------------------------------------------#

def _patch_target(patch_key: Any) -> str:
    """Return the model state-dict key a patch entry targets.

    ``load_lora`` keys are either the key itself or ``(key, offset)`` for
    patches into a slice of a fused weight — the same shape ``ModelPatcher.
    add_patches`` unpacks.
    """
    return patch_key if isinstance(patch_key, str) else patch_key[0]


def _build_key_maps(model, clip) -> Tuple[Dict[str, Any], set]:
    """Build ComfyUI's LoRA key map and the set of keys owned by the UNet.

    Returning the UNet key set is what lets us route text-encoder patches to the
    CLIP patcher and everything else to the model patcher without guessing from
    key names.
    """
    unet_map: Dict[str, Any] = {}
    clip_map: Dict[str, Any] = {}

    if model is not None and hasattr(comfy.lora, "model_lora_keys_unet"):
        comfy.lora.model_lora_keys_unet(model.model, unet_map)
    if clip is not None and hasattr(comfy.lora, "model_lora_keys_clip"):
        comfy.lora.model_lora_keys_clip(clip.cond_stage_model, clip_map)

    key_map = dict(clip_map)
    key_map.update(unet_map)  # UNet wins on the (unlikely) collision
    unet_targets = {_patch_target(v) for v in unet_map.values()}
    return key_map, unet_targets


def _flux_geometry(model) -> Tuple[int, int]:
    """Read the double/single stream depths off the loaded FLUX model."""
    try:
        config = model.model.model_config.unet_config
        depth = int(config.get("depth", 0)) or FLUX_DEFAULT_DEPTH
        depth_single = int(config.get("depth_single_blocks", 0)) or FLUX_DEFAULT_DEPTH_SINGLE
        return depth, depth_single
    except Exception:  # noqa: BLE001 - any unexpected config shape falls back
        return FLUX_DEFAULT_DEPTH, FLUX_DEFAULT_DEPTH_SINGLE


def _format_report(tag: str,
                   lora_name: str,
                   preset: str,
                   blocks: List[str],
                   grouped: Dict[str, Dict[Any, Any]],
                   strengths: Dict[str, float],
                   applied: Dict[str, int]) -> str:
    lines = [f"[{tag}] {lora_name}  (preset: {preset})",
             f"{'block':<40} {'weight':>7} {'found':>7} {'applied':>8}"]
    for name in blocks:
        lines.append(
            f"{name:<40} {strengths.get(name, 0.0):>7.2f} "
            f"{len(grouped.get(name, {})):>7} {applied.get(name, 0):>8}"
        )
    total_found = sum(len(g) for g in grouped.values())
    total_applied = sum(applied.values())
    lines.append(f"{'TOTAL':<40} {'':>7} {total_found:>7} {total_applied:>8}")
    return "\n".join(lines)


def _explain_empty_blocks(tag: str,
                          blocks: List[str],
                          grouped: Dict[str, Dict[Any, Any]],
                          strengths: Dict[str, float]) -> None:
    """Log why a block with a non-zero weight contributed nothing."""
    for name in blocks:
        if grouped.get(name) or strengths.get(name, 0.0) == 0.0:
            continue
        logger.info(
            "[%s] %s: weight %.2f but this LoRA contains no tensors for that block.",
            tag, name, strengths[name],
        )


# -----------------------------------------------------------------------------#
#                                 SHARED  NODE                                 #
# -----------------------------------------------------------------------------#

class _BobsLoraLoaderBase:
    """Shared load / classify / patch pipeline for the FLUX and SDXL loaders."""

    FAMILY = "FLUX"
    BLOCKS: List[str] = ALL_FLUX_BLOCKS

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    OUTPUT_TOOLTIPS = (
        "Model with the block-weighted LoRA applied.",
        "CLIP with the text-encoder portion of the LoRA applied.",
        "Per-block report: weight, tensors found and tensors actually patched.",
    )
    FUNCTION = "apply_lora"
    CATEGORY = "Bobs_Nodes"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache_key: Optional[tuple] = None
        self._cache_sd: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- inputs --

    @classmethod
    def _input_types(cls) -> Dict[str, Any]:
        required = {
            "model": ("MODEL", {"tooltip": "Diffusion model to patch."}),
            "lora_name": (
                ["None"] + folder_paths.get_filename_list("loras"),
                {"tooltip": "LoRA file to load. 'None' passes the model through untouched."},
            ),
            "strength": ("FLOAT", {
                "default": 1.0, "min": -MAX_STRENGTH, "max": MAX_STRENGTH, "step": 0.01,
                "tooltip": "Global multiplier applied on top of every block weight.",
            }),
            "preset": (
                list(LORA_BLOCK_PRESETS[cls.FAMILY].keys()),
                {"tooltip": "Choose 'Custom' to use the sliders below; any other preset overrides them."},
            ),
        }
        for block in cls.BLOCKS:
            required[block] = ("FLOAT", {
                "default": 1.0, "min": -MAX_BLOCK_WEIGHT, "max": MAX_BLOCK_WEIGHT, "step": 0.05,
                "tooltip": BLOCK_TOOLTIPS.get(block, block),
            })
        return {
            "required": required,
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional. Leave unconnected to patch the model only."}),
            },
        }

    # ------------------------------------------------------------ lora  file --

    def _load_lora_state_dict(self, lora_path: str) -> Dict[str, Any]:
        """Load a LoRA, reusing the previous load when the file is unchanged."""
        try:
            stat = os.stat(lora_path)
            cache_key = (lora_path, stat.st_mtime_ns, stat.st_size)
        except OSError:
            cache_key = None

        if cache_key is not None and cache_key == self._cache_key:
            return self._cache_sd

        # load_torch_file handles .safetensors and pickled checkpoints, and
        # applies weights_only=True to the latter.
        state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        if _lora_convert is not None:
            state_dict = _lora_convert.convert_lora(state_dict)

        self._cache_key, self._cache_sd = cache_key, state_dict
        return state_dict

    # ------------------------------------------------------------ classifier --

    def _classifier(self, model):
        """Return ``fn(model_state_dict_key) -> block name`` for this family."""
        raise NotImplementedError

    # ----------------------------------------------------------------- apply --

    def apply_lora(self, model, lora_name, strength, preset, clip=None, **kwargs):
        tag = self.FAMILY

        if lora_name in (None, "", "None"):
            return (model, clip, f"[{tag}] no LoRA selected.")
        if strength == 0.0:
            return (model, clip, f"[{tag}] {lora_name}: strength is 0.00, nothing applied.")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path:
            message = f"[{tag}] LoRA file not found: {lora_name}"
            self.logger.error(message)
            return (model, clip, message)

        try:
            lora_sd = self._load_lora_state_dict(lora_path)
        except Exception as exc:  # noqa: BLE001 - surface, never crash the graph
            message = f"[{tag}] failed to read {lora_name}: {exc}"
            self.logger.error(message)
            return (model, clip, message)

        strengths = resolve_block_strengths(self.FAMILY, preset, strength, kwargs)

        key_map, unet_targets = _build_key_maps(model, clip)
        if not key_map:
            message = f"[{tag}] could not build a key map for this model; LoRA not applied."
            self.logger.warning(message)
            return (model, clip, message)

        all_patches = comfy.lora.load_lora(lora_sd, key_map)
        if not all_patches:
            message = (f"[{tag}] {lora_name}: none of its tensors match this model. "
                       f"Is it a LoRA for a different architecture?")
            self.logger.warning(message)
            return (model, clip, message)

        # Group UNet patches per conceptual block; CLIP patches all share the
        # text-encoder block.
        classify = self._classifier(model)
        text_encoder_block = TEXT_ENCODER_BLOCK[self.FAMILY]
        grouped: Dict[str, Dict[Any, Any]] = {name: {} for name in self.BLOCKS}

        for patch_key, patch in all_patches.items():
            target = _patch_target(patch_key)
            block = classify(target) if target in unet_targets else text_encoder_block
            grouped[block][patch_key] = patch

        out_model = model.clone() if model is not None else None
        out_clip = clip.clone() if clip is not None else None

        applied: Dict[str, int] = {}
        for name in self.BLOCKS:
            patches = grouped[name]
            block_strength = strengths.get(name, 0.0)
            patcher = out_clip if name == text_encoder_block else out_model
            if not patches or block_strength == 0.0 or patcher is None:
                applied[name] = 0
                continue
            applied[name] = len(patcher.add_patches(patches, block_strength))

        # Any tensor in the file that ComfyUI could not place is reported by
        # comfy.lora.load_lora itself as a "lora key not loaded" warning.
        report = _format_report(tag, lora_name, preset, self.BLOCKS,
                                grouped, strengths, applied)
        self.logger.info("\n%s", report)
        _explain_empty_blocks(tag, self.BLOCKS, grouped, strengths)

        return (out_model, out_clip, report)


# -----------------------------------------------------------------------------#
#                               FLUX  LOADER                                   #
# -----------------------------------------------------------------------------#

class BobsLoraLoaderFlux(_BobsLoraLoaderBase):
    FAMILY = "FLUX"
    BLOCKS = ALL_FLUX_BLOCKS
    DESCRIPTION = (
        "Applies a LoRA to a FLUX model with an independent weight per conceptual "
        "block (composition, subject, style core, detail, texture, text encoder). "
        "Block ranges are derived from the loaded model's own depth, so FLUX.1 "
        "dev/schnell and pruned variants are all handled."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return cls._input_types()

    def _classifier(self, model):
        ranges = flux_block_ranges(*_flux_geometry(model))
        return lambda key: classify_flux_key(key, ranges)


# -----------------------------------------------------------------------------#
#                               SDXL  LOADER                                   #
# -----------------------------------------------------------------------------#

class BobsLoraLoaderSdxl(_BobsLoraLoaderBase):
    FAMILY = "SDXL"
    BLOCKS = ALL_SDXL_BLOCKS
    DESCRIPTION = (
        "Applies a LoRA to an SDXL model with independent weights for the text "
        "encoder and the UNet input / middle / output stages."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return cls._input_types()

    def _classifier(self, model):
        return classify_sdxl_key


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
