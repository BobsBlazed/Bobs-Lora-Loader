"""Unit tests for the block-classification logic.

These run without ComfyUI or torch installed:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bobs_blocks import (  # noqa: E402
    ALL_BLOCKS,
    ALL_FLUX_BLOCKS,
    ALL_SDXL_BLOCKS,
    FLUX_CORE,
    FLUX_DEFAULT_DEPTH,
    FLUX_DEFAULT_DEPTH_SINGLE,
    FLUX_EARLY_DOWN,
    FLUX_EARLY_UP,
    FLUX_FINAL,
    FLUX_GUIDANCE,
    FLUX_IMAGE_HINT,
    FLUX_LATE_DOWN,
    FLUX_LATE_UP,
    FLUX_MID_DOWN,
    FLUX_MID_UP,
    FLUX_OTHER,
    FLUX_TEXT_CONDITIONING,
    FLUX_TEXT_ENCODER,
    FLUX_TIMESTEP,
    FLUX_VECTOR,
    LORA_BLOCK_PRESETS,
    SDXL_INPUT_BLOCKS,
    SDXL_MIDDLE_BLOCK,
    SDXL_OTHER,
    SDXL_OUTPUT_BLOCKS,
    SDXL_TEXT_ENCODER,
    classify_flux_key,
    classify_sdxl_key,
    flux_block_ranges,
    normalize_key,
    resolve_block_strengths,
)


class TestNormalizeKey(unittest.TestCase):
    def test_separators_collapse_to_underscores(self):
        self.assertEqual(
            normalize_key("diffusion_model.double_blocks.3.img_attn.qkv.weight"),
            "_diffusion_model_double_blocks_3_img_attn_qkv_weight",
        )

    def test_leading_underscore_is_always_present(self):
        self.assertTrue(normalize_key("txt_in.weight").startswith("_txt_in"))
        self.assertTrue(normalize_key("_already").startswith("_already"))

    def test_runs_of_separators_collapse(self):
        self.assertEqual(normalize_key("a..b//c"), "_a_b_c")


class TestFluxRanges(unittest.TestCase):
    def test_canonical_geometry_reproduces_documented_ranges(self):
        double, single = flux_block_ranges(FLUX_DEFAULT_DEPTH, FLUX_DEFAULT_DEPTH_SINGLE)
        self.assertEqual(double, [(4, FLUX_EARLY_DOWN), (8, FLUX_MID_DOWN),
                                  (10, FLUX_LATE_DOWN), (19, FLUX_CORE)])
        self.assertEqual(single, [(8, FLUX_CORE), (16, FLUX_EARLY_UP),
                                  (32, FLUX_MID_UP), (38, FLUX_LATE_UP)])

    def test_boundaries_are_monotonic_and_cover_the_stack(self):
        for depth in range(0, 60):
            double, single = flux_block_ranges(depth, depth)
            for bounds in (double, single):
                uppers = [u for u, _ in bounds]
                self.assertEqual(uppers, sorted(uppers), f"depth={depth}")
                self.assertEqual(uppers[-1], depth, f"depth={depth}")

    def test_every_index_of_a_nonstandard_depth_is_classified(self):
        ranges = flux_block_ranges(8, 16)
        for i in range(8):
            key = f"diffusion_model.double_blocks.{i}.img_attn.qkv.weight"
            self.assertIn(classify_flux_key(key, ranges), ALL_FLUX_BLOCKS)
            self.assertNotEqual(classify_flux_key(key, ranges), FLUX_OTHER)
        for i in range(16):
            key = f"diffusion_model.single_blocks.{i}.linear1.weight"
            self.assertNotEqual(classify_flux_key(key, ranges), FLUX_OTHER)


class TestFluxClassification(unittest.TestCase):
    def test_double_block_index_ranges(self):
        cases = [(0, FLUX_EARLY_DOWN), (3, FLUX_EARLY_DOWN),
                 (4, FLUX_MID_DOWN), (7, FLUX_MID_DOWN),
                 (8, FLUX_LATE_DOWN), (9, FLUX_LATE_DOWN),
                 (10, FLUX_CORE), (18, FLUX_CORE)]
        for index, expected in cases:
            key = f"diffusion_model.double_blocks.{index}.img_mlp.0.weight"
            self.assertEqual(classify_flux_key(key), expected, key)

    def test_single_block_index_ranges(self):
        cases = [(0, FLUX_CORE), (7, FLUX_CORE),
                 (8, FLUX_EARLY_UP), (15, FLUX_EARLY_UP),
                 (16, FLUX_MID_UP), (31, FLUX_MID_UP),
                 (32, FLUX_LATE_UP), (37, FLUX_LATE_UP)]
        for index, expected in cases:
            key = f"diffusion_model.single_blocks.{index}.linear1.weight"
            self.assertEqual(classify_flux_key(key), expected, key)

    def test_single_is_not_shadowed_by_the_double_pattern(self):
        # "single_transformer_blocks_3" also contains "transformer_blocks_3".
        self.assertEqual(
            classify_flux_key("single_transformer_blocks.3.attn.to_q.weight"),
            FLUX_CORE,
        )
        self.assertEqual(
            classify_flux_key("transformer_blocks.3.attn.to_q.weight"),
            FLUX_EARLY_DOWN,
        )

    def test_head_and_tail_tokens(self):
        cases = {
            "diffusion_model.txt_in.weight": FLUX_TEXT_CONDITIONING,
            "diffusion_model.time_in.in_layer.weight": FLUX_TIMESTEP,
            "diffusion_model.img_in.weight": FLUX_IMAGE_HINT,
            "diffusion_model.guidance_in.in_layer.weight": FLUX_GUIDANCE,
            "diffusion_model.vector_in.in_layer.weight": FLUX_VECTOR,
            "diffusion_model.final_layer.linear.weight": FLUX_FINAL,
            "diffusion_model.final_layer.adaLN_modulation.1.weight": FLUX_FINAL,
        }
        for key, expected in cases.items():
            self.assertEqual(classify_flux_key(key), expected, key)

    def test_diffusers_head_tokens(self):
        cases = {
            "time_text_embed.guidance_embedder.linear_1.weight": FLUX_GUIDANCE,
            "time_text_embed.text_embedder.linear_1.weight": FLUX_VECTOR,
            "time_text_embed.timestep_embedder.linear_1.weight": FLUX_TIMESTEP,
            "context_embedder.weight": FLUX_TEXT_CONDITIONING,
            "x_embedder.weight": FLUX_IMAGE_HINT,
            "proj_out.weight": FLUX_FINAL,
        }
        for key, expected in cases.items():
            self.assertEqual(classify_flux_key(key), expected, key)

    def test_block_keys_win_over_head_tokens(self):
        # img_attn / txt_attn inside a block must not be read as img_in / txt_in.
        self.assertEqual(
            classify_flux_key("diffusion_model.double_blocks.0.img_attn.proj.weight"),
            FLUX_EARLY_DOWN,
        )
        self.assertEqual(
            classify_flux_key("diffusion_model.double_blocks.0.txt_mod.lin.weight"),
            FLUX_EARLY_DOWN,
        )

    def test_unknown_key_falls_through_to_other(self):
        self.assertEqual(classify_flux_key("diffusion_model.mystery.weight"), FLUX_OTHER)

    def test_whole_flux_state_dict_shape_is_covered(self):
        """No canonical FLUX.1 UNet key should land in 'Other Tensors'."""
        keys = [
            "diffusion_model.img_in.weight",
            "diffusion_model.txt_in.weight",
            "diffusion_model.time_in.in_layer.weight",
            "diffusion_model.time_in.out_layer.weight",
            "diffusion_model.vector_in.in_layer.weight",
            "diffusion_model.guidance_in.in_layer.weight",
            "diffusion_model.final_layer.linear.weight",
        ]
        for i in range(FLUX_DEFAULT_DEPTH):
            keys += [
                f"diffusion_model.double_blocks.{i}.img_attn.qkv.weight",
                f"diffusion_model.double_blocks.{i}.txt_attn.proj.weight",
                f"diffusion_model.double_blocks.{i}.img_mod.lin.weight",
                f"diffusion_model.double_blocks.{i}.txt_mlp.2.weight",
            ]
        for i in range(FLUX_DEFAULT_DEPTH_SINGLE):
            keys += [
                f"diffusion_model.single_blocks.{i}.linear1.weight",
                f"diffusion_model.single_blocks.{i}.linear2.weight",
                f"diffusion_model.single_blocks.{i}.modulation.lin.weight",
            ]
        unclassified = [k for k in keys if classify_flux_key(k) == FLUX_OTHER]
        self.assertEqual(unclassified, [])


class TestSdxlClassification(unittest.TestCase):
    def test_unet_stages(self):
        cases = {
            "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight":
                SDXL_INPUT_BLOCKS,
            "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight":
                SDXL_MIDDLE_BLOCK,
            "diffusion_model.output_blocks.5.1.proj_out.weight":
                SDXL_OUTPUT_BLOCKS,
            "diffusion_model.time_embed.0.weight": SDXL_OTHER,
            "diffusion_model.label_emb.0.0.weight": SDXL_OTHER,
            "diffusion_model.out.2.weight": SDXL_OTHER,
        }
        for key, expected in cases.items():
            self.assertEqual(classify_sdxl_key(key), expected, key)

    def test_nested_transformer_blocks_do_not_confuse_the_stage(self):
        key = "diffusion_model.output_blocks.3.1.transformer_blocks.1.ff.net.0.proj.weight"
        self.assertEqual(classify_sdxl_key(key), SDXL_OUTPUT_BLOCKS)


class TestNestedStackIsolation(unittest.TestCase):
    """The dedicated classifiers must not read a *nested* block stack.

    SDXL nests transformer_blocks inside input_blocks/middle_block/
    output_blocks. Matching on the underscore-normalised key made the FLUX
    classifier see that inner stack and file SDXL keys as double-stream blocks.
    """

    SDXL_KEYS = [
        "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight",
        "diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight",
        "diffusion_model.output_blocks.5.1.transformer_blocks.1.ff.net.0.proj.weight",
    ]

    def test_flux_classifier_does_not_claim_sdxl_keys(self):
        for key in self.SDXL_KEYS:
            self.assertEqual(classify_flux_key(key), FLUX_OTHER, key)

    def test_sdxl_classifier_still_reads_the_outer_stage(self):
        expected = [SDXL_INPUT_BLOCKS, SDXL_MIDDLE_BLOCK, SDXL_OUTPUT_BLOCKS]
        for key, want in zip(self.SDXL_KEYS, expected):
            self.assertEqual(classify_sdxl_key(key), want, key)

    def test_flux_keys_are_unaffected(self):
        self.assertEqual(
            classify_flux_key("diffusion_model.double_blocks.0.img_attn.qkv.weight"),
            FLUX_EARLY_DOWN)
        self.assertEqual(
            classify_flux_key("diffusion_model.single_blocks.37.linear1.weight"),
            FLUX_LATE_UP)
        # diffusers spellings still resolve
        self.assertEqual(classify_flux_key("transformer_blocks.3.attn.to_q.weight"),
                         FLUX_EARLY_DOWN)
        self.assertEqual(classify_flux_key("single_transformer_blocks.3.attn.to_q.weight"),
                         FLUX_CORE)

    def test_flux_negative_index_resolves_against_the_stack(self):
        self.assertEqual(
            classify_flux_key("diffusion_model.double_blocks.-1.img_attn.qkv.weight"),
            FLUX_CORE)


class TestStrengthResolution(unittest.TestCase):
    def test_custom_preset_uses_sliders_scaled_by_global_strength(self):
        overrides = {name: 0.5 for name in ALL_SDXL_BLOCKS}
        result = resolve_block_strengths("SDXL", "Custom", 2.0, overrides)
        self.assertEqual(set(result), set(ALL_SDXL_BLOCKS))
        for value in result.values():
            self.assertAlmostEqual(value, 1.0)

    def test_missing_slider_defaults_to_one(self):
        result = resolve_block_strengths("SDXL", "Custom", 1.0, {})
        self.assertAlmostEqual(result[SDXL_INPUT_BLOCKS], 1.0)

    def test_named_preset_ignores_sliders(self):
        overrides = {name: 0.0 for name in ALL_SDXL_BLOCKS}
        result = resolve_block_strengths("SDXL", "Style", 1.0, overrides)
        self.assertAlmostEqual(result[SDXL_OUTPUT_BLOCKS], 1.0)
        self.assertAlmostEqual(result[SDXL_TEXT_ENCODER], 0.0)

    def test_preset_strength_multiplies_global_strength(self):
        # "Fix Hands/Anatomy" carries a preset strength of 0.4.
        result = resolve_block_strengths("SDXL", "Fix Hands/Anatomy", 0.5, {})
        self.assertAlmostEqual(result[SDXL_INPUT_BLOCKS], 0.2)

    def test_negative_global_strength_inverts_every_block(self):
        result = resolve_block_strengths("FLUX", "Full (Normal LoRA)", -1.0, {})
        for value in result.values():
            self.assertAlmostEqual(value, -1.0)

    def test_full_preset_is_equivalent_to_a_plain_lora_load(self):
        for family in ("FLUX", "SDXL"):
            result = resolve_block_strengths(family, "Full (Normal LoRA)", 0.8, {})
            for name, value in result.items():
                self.assertAlmostEqual(value, 0.8, msg=f"{family}/{name}")


class TestPresetIntegrity(unittest.TestCase):
    def test_every_preset_covers_every_block(self):
        # Drive off ALL_BLOCKS rather than a literal family map: other modules
        # register additional families, and this test must not care whether
        # they happen to have been imported first.
        for family, presets in LORA_BLOCK_PRESETS.items():
            blocks = ALL_BLOCKS[family]
            for preset_name, config in presets.items():
                if preset_name == "Custom":
                    self.assertEqual(config, {})
                    continue
                weights = config["block_weights"]
                self.assertEqual(set(weights), set(blocks),
                                 f"{family}/{preset_name}")

    def test_block_lists_have_no_duplicates(self):
        for blocks in (ALL_FLUX_BLOCKS, ALL_SDXL_BLOCKS):
            self.assertEqual(len(blocks), len(set(blocks)))

    def test_widget_order_contract_is_preserved(self):
        """Widget values are serialised positionally; only append to these."""
        self.assertEqual(ALL_FLUX_BLOCKS[:14], [
            "Text Conditioning",
            "Timestep Embedding",
            "Image Hint",
            "Guidance Embedding",
            "Vector Embedding",
            "Early Downsampling (Composition)",
            "Mid Downsampling (Subject & Concept)",
            "Late Downsampling (Refinement)",
            "Core/Middle Block (Style Focus)",
            "Early Upsampling (Initial Style)",
            "Mid Upsampling (Detail Generation)",
            "Late Upsampling (Final Textures)",
            "Final Output Layer (Latent Projection)",
            "Other Tensors",
        ])
        self.assertEqual(ALL_FLUX_BLOCKS[14], FLUX_TEXT_ENCODER)
        self.assertEqual(ALL_SDXL_BLOCKS[:4], [
            "Text Encoder",
            "Input Blocks",
            "Middle Block",
            "Output Blocks",
        ])
        self.assertEqual(ALL_SDXL_BLOCKS[4], SDXL_OTHER)


if __name__ == "__main__":
    unittest.main()
