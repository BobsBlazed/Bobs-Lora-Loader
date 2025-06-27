# Bobs LoRA Loader for ComfyUI

An advanced LoRA loader for ComfyUI that provides granular, block-level control over how a LoRA is applied to both **SDXL** and **FLUX** models.

This node allows you to go beyond a single strength slider and specify different weights for distinct parts of the model, such as the text encoder, the U-Net input blocks, and the output blocks. This is particularly useful for mixing and matching LoRA concepts, strengthening character details while reducing stylistic influence, or vice-versa.

## Features

-   **Dual Model Support**: Separate, optimized loaders for `SDXL` and `FLUX` models.
-   **Block-Weighting**: Fine-tune the strength of a LoRA on different conceptual parts of the diffusion model.
-   **Easy-to-Use Presets**: Comes with pre-configured presets for common use cases like `Character`, `Style`, `Concept`, and `Fix Hands/Anatomy`.
-   **Custom Mode**: Disable presets to get direct slider control over every block for ultimate customization.
-   **Normal LoRA Functionality**: To use it like a standard LoRA loader, simply select the `Full (Normal LoRA)` preset.

## Installation

1.  Navigate to your ComfyUI `custom_nodes` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  Clone this repository:
    ```bash
    git clone [URL_OF_YOUR_GITHUB_REPO]
    ```
3.  Restart ComfyUI.

## How to Use

1.  In ComfyUI, add the node by right-clicking, selecting "Add Node," and navigating to the `Bobs_Nodes` category.
2.  Choose either **Bobs LoRA Loader (SDXL)** or **Bobs LoRA Loader (FLUX)** depending on your base model.
3.  Connect your `MODEL` and `CLIP` outputs into the corresponding inputs on the node.
4.  Select the LoRA you wish to apply from the `lora_name` dropdown.
5.  Use the `preset` dropdown to quickly apply a set of block weights for a specific purpose (e.g., "Character").
6.  If you want to fine-tune the weights, set the `preset` to `Custom` and adjust the individual block sliders that appear.
7.  The main `strength` slider acts as a global multiplier on top of the preset or custom weights.

## Why Use Block-Weighted LoRA?

A single LoRA file often contains training for multiple concepts (e.g., a character's face, their clothing, and the overall artistic style). A standard LoRA loader applies the LoRA with one uniform strength across the entire model.

This can be limiting. For example:
-   You might want the character's features but not the stiff, overbaked style it was trained with.
-   You might want a LoRA's artistic style but not the character concepts embedded within it.

By assigning different strengths to different model blocks, you can selectively emphasize or de-emphasize these aspects. The `FLUX` loader provides even more granular control, breaking the model down into conceptual stages like "Composition," "Subject & Concept," and "Final Textures."

## Bug Fixes in This Version

This version includes a critical fix for the **BobsLoraLoaderFlux** node, where block weights and presets were not being applied correctly, causing the node to produce the same output regardless of settings. The patch application logic has been completely rewritten to ensure all sliders and presets work as intended.
