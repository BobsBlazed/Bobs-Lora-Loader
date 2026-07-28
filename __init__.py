"""Bobs LoRA Loader — block-weighted LoRA loading for ComfyUI (FLUX + SDXL)."""

import logging

from .bobs_lora_loader import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    BobsLoraLoaderFlux,
    BobsLoraLoaderSdxl,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "BobsLoraLoaderFlux",
    "BobsLoraLoaderSdxl",
]

logging.getLogger("BobsLoraLoader").info(
    "Bobs LoRA Loader: %d nodes registered.", len(NODE_CLASS_MAPPINGS)
)
