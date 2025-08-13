# Bobs LoRA Loader for ComfyUI

|||
|---|---|
|![image](https://github.com/user-attachments/assets/f614b579-c232-4f33-b994-f196c225edcf)|![image](https://github.com/user-attachments/assets/fca84c9b-211e-41fc-86a9-583e187cd6f1)|

An advanced LoRA loader for ComfyUI that provides granular, block-level control over how a LoRA is applied to both **SDXL** and **FLUX** models, giving you unparalleled control over your image generation process.

This node allows you to go beyond a single strength slider and specify different weights for distinct parts of the model, such as the text encoder, the U-Net input blocks, and the output blocks. This is particularly useful for mixing and matching LoRA concepts, strengthening character details while reducing stylistic influence, or vice-versa.

## Features

-   **Dual Model Support**: Separate, optimized loaders for `SDXL` and `FLUX` models, each tailored to the architecture's specific blocks.
-   **Granular Block-Level Control**: Fine-tune the strength of a LoRA on different conceptual parts of the diffusion model.
-   **Intelligent Presets**: Comes with pre-configured presets for common use cases like `Character`, `Style`, `Concept`, and `Fix Hands/Anatomy`.
-   **Full Customization**: Set the `preset` to `Custom` to get direct slider control over every block for ultimate fine-tuning.
-   **Robust LoRA Compatibility**: Intelligently handles multiple LoRA naming conventions (e.g., `lora_unet_...`, `unet...`, `lora_transformer_...`, etc.) to ensure broad compatibility with community-made files.
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
3.  Connect your `MODEL` and `CLIP` outputs into the corresponding inputs on the node.
4.  Select the LoRA you wish to apply from the `lora_name` dropdown.
5.  Use the `preset` dropdown to quickly apply a set of block weights for a specific purpose (e.g., "Character" to focus on subject detail).
6.  For maximum control, set the `preset` to `Custom` and adjust the individual block sliders that appear.
7.  The main `strength` slider acts as a global multiplier for all other block weights, allowing you to scale the entire effect up or down.

## Why Use Block-Weighted LoRA?

A single LoRA file often contains training for multiple concepts (e.g., a character's face, their clothing, and the overall artistic style). A standard LoRA loader applies the LoRA with one uniform strength across the entire model.

This can be limiting. For example:
-   You might want a character's features but not the stiff, overbaked style it was trained with.
-   You might want a LoRA's artistic style but not the character concepts embedded within it.

By assigning different strengths to different model blocks, you can selectively emphasize or de-emphasize these aspects. The SDXL loader provides coarse control over the main UNet stages, while the FLUX loader offers even finer-grained control over conceptual phases like "Composition," "Refinement," and "Final Textures."

## Changelog / What's New

### Major Overhaul of LoRA Classification Logic

This version addresses a critical issue where many LoRA files were not being correctly categorized, causing a large number of weights to be ignored or misapplied. The core classification logic for both loaders has been completely rewritten.

-   **FLUX Loader Fix**: The loader's regular expressions and classification rules have been completely overhauled to correctly handle all block types, including complex `time_text_embed` keys and all layers within attention blocks (`q`, `k`, `v`, `out`).
-   **SDXL Loader Fix**: The loader is now compatible with LoRAs that use the `unet.` naming convention (common in some community models), in addition to the standard `lora_unet_...` format. This significantly improves compatibility.

The result is a far more reliable and robust tool that ensures the settings on your sliders accurately reflect the changes being applied to the model. The "Other Tensors" count in the logs should now be zero for all supported LoRAs.
