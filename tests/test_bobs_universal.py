"""Unit tests for the architecture-agnostic block analysis.

Runs without ComfyUI or torch:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The package modules use relative imports; expose them under a package name.
import importlib.util  # noqa: E402
import types  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("bobs_pkg")
_pkg.__path__ = [_REPO]
sys.modules.setdefault("bobs_pkg", _pkg)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"bobs_pkg.{name}", os.path.join(_REPO, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"bobs_pkg.{name}"] = module
    spec.loader.exec_module(module)
    return module


bobs_blocks = _load("bobs_blocks")
bu = _load("bobs_universal")


# --------------------------------------------------------------------------- #
# Representative state-dict key sets, one per architecture family.
# Names follow ComfyUI's canonical "diffusion_model.*" naming.
# --------------------------------------------------------------------------- #

def _unet_sd(n_in=9, n_out=9):
    keys = ["diffusion_model.time_embed.0.weight",
            "diffusion_model.label_emb.0.0.weight",
            "diffusion_model.out.2.weight"]
    for i in range(n_in):
        keys.append(f"diffusion_model.input_blocks.{i}.1.transformer_blocks.0.attn1.to_q.weight")
    keys.append("diffusion_model.middle_block.1.transformer_blocks.0.attn2.to_k.weight")
    for i in range(n_out):
        keys.append(f"diffusion_model.output_blocks.{i}.1.transformer_blocks.0.attn1.to_v.weight")
    return keys


def _flat_stack(stack, count, extra=()):
    keys = [f"diffusion_model.{stack}.{i}.attn.to_q.weight" for i in range(count)]
    return keys + [f"diffusion_model.{k}" for k in extra]


def _dual_stack(first, second, n1, n2, extra=()):
    keys = [f"diffusion_model.{first}.{i}.attn.qkv.weight" for i in range(n1)]
    keys += [f"diffusion_model.{second}.{i}.linear1.weight" for i in range(n2)]
    return keys + [f"diffusion_model.{k}" for k in extra]


FAMILIES = {
    "SD15": _unet_sd(12, 12),
    "SDXL": _unet_sd(9, 9),
    "FLUX": _dual_stack("double_blocks", "single_blocks", 19, 38,
                        extra=("img_in.weight", "txt_in.weight",
                               "time_in.in_layer.weight", "guidance_in.in_layer.weight",
                               "vector_in.in_layer.weight", "final_layer.linear.weight")),
    "HiDream": _dual_stack("double_stream_blocks", "single_stream_blocks", 16, 32,
                           extra=("x_embedder.weight", "final_layer.linear.weight")),
    "HunyuanVideo": _dual_stack("double_blocks", "single_blocks", 20, 40,
                                extra=("img_in.proj.weight", "final_layer.linear.weight")),
    "SD3": _flat_stack("joint_blocks", 24,
                       extra=("x_embedder.proj.weight", "t_embedder.mlp.0.weight",
                              "y_embedder.mlp.0.weight", "context_embedder.weight",
                              "pos_embed", "final_layer.linear.weight")),
    "AuraFlow": (_flat_stack("joint_transformer_blocks", 4)
                 + _flat_stack("single_transformer_blocks", 32)
                 + ["diffusion_model.init_x_linear.weight",
                    "diffusion_model.final_linear.weight"]),
    "QwenImage": _flat_stack("transformer_blocks", 60,
                             extra=("img_in.weight", "txt_in.weight",
                                    "time_text_embed.timestep_embedder.linear_1.weight",
                                    "proj_out.weight")),
    "LTXV": _flat_stack("transformer_blocks", 28,
                        extra=("patchify_proj.weight", "caption_projection.linear_1.weight",
                               "proj_out.weight")),
    "PixArt": _flat_stack("blocks", 28,
                          extra=("x_embedder.proj.weight", "t_embedder.mlp.0.weight",
                                 "final_layer.linear.weight")),
    "Wan21": _flat_stack("blocks", 40,
                         extra=("patch_embedding.weight", "time_embedding.0.weight",
                                "text_embedding.0.weight", "head.head.weight")),
    "Mochi": _flat_stack("blocks", 48,
                         extra=("x_embedder.proj.weight", "final_layer.linear.weight")),
    "Lumina2": _flat_stack("layers", 26,
                           extra=("x_embedder.weight", "cap_embedder.1.weight",
                                  "final_layer.linear.weight")),
    "Cosmos": _flat_stack("blocks", 28,
                          extra=("x_embedder.proj.1.weight", "final_layer.linear.weight")),
}


class TestStackDiscovery(unittest.TestCase):
    def test_discovers_expected_stacks_and_sizes(self):
        cases = {
            "SD15": {"input_blocks": 12, "middle_block": 1, "output_blocks": 12},
            "SDXL": {"input_blocks": 9, "middle_block": 1, "output_blocks": 9},
            "FLUX": {"double_blocks": 19, "single_blocks": 38},
            "HiDream": {"double_stream_blocks": 16, "single_stream_blocks": 32},
            "SD3": {"joint_blocks": 24},
            "AuraFlow": {"joint_transformer_blocks": 4, "single_transformer_blocks": 32},
            "QwenImage": {"transformer_blocks": 60},
            "Wan21": {"blocks": 40},
            "Lumina2": {"layers": 26},
        }
        for family, expected in cases.items():
            layout = bu.discover_layout(FAMILIES[family])
            self.assertEqual(layout.stacks, expected, family)

    def test_stacks_are_ordered_by_execution_order(self):
        layout = bu.discover_layout(FAMILIES["SD15"])
        self.assertEqual(list(layout.stacks), ["input_blocks", "middle_block", "output_blocks"])
        layout = bu.discover_layout(FAMILIES["FLUX"])
        self.assertEqual(list(layout.stacks), ["double_blocks", "single_blocks"])
        layout = bu.discover_layout(FAMILIES["AuraFlow"])
        self.assertEqual(list(layout.stacks),
                         ["joint_transformer_blocks", "single_transformer_blocks"])

    def test_outermost_stack_wins_over_nested_one(self):
        # SDXL nests transformer_blocks inside input_blocks; the outer one counts.
        key = "diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight"
        self.assertEqual(bu._stack_match(key), ("input_blocks", 4))

    def test_longer_stack_name_wins(self):
        for key, expected in [
            ("diffusion_model.single_transformer_blocks.3.x.weight",
             ("single_transformer_blocks", 3)),
            ("diffusion_model.single_blocks.3.x.weight", ("single_blocks", 3)),
            ("diffusion_model.double_stream_blocks.2.x.weight", ("double_stream_blocks", 2)),
            ("diffusion_model.blocks.7.x.weight", ("blocks", 7)),
            ("diffusion_model.layers.7.x.weight", ("layers", 7)),
        ]:
            self.assertEqual(bu._stack_match(key), expected, key)

    def test_unknown_stack_name_still_discovered(self):
        keys = [f"diffusion_model.mystery_blocks.{i}.attn.weight" for i in range(10)]
        layout = bu.discover_layout(keys)
        self.assertEqual(layout.stacks, {"mystery_blocks": 10})
        self.assertEqual(layout.total, 10)

    def test_empty_layout_is_falsy_and_described(self):
        layout = bu.discover_layout(["diffusion_model.final_layer.linear.weight"])
        self.assertFalse(layout)
        self.assertIn("no block stacks", layout.describe())


class TestUniversalClassification(unittest.TestCase):
    def test_every_family_covers_its_whole_stack_without_other(self):
        """No key from any family should fall through to 'Other Tensors'."""
        for family, keys in FAMILIES.items():
            layout = bu.discover_layout(keys)
            unclassified = [k for k in keys
                            if bu.classify_universal_key(k, layout) == bu.UNIVERSAL_OTHER]
            self.assertEqual(unclassified, [], f"{family}: {unclassified[:5]}")

    def test_every_depth_bucket_is_reachable(self):
        for family, keys in FAMILIES.items():
            layout = bu.discover_layout(keys)
            if not layout:
                continue
            buckets = {bu.classify_universal_key(k, layout) for k in keys}
            for expected in (bu.UNIVERSAL_EARLY, bu.UNIVERSAL_MID, bu.UNIVERSAL_LATE):
                self.assertIn(expected, buckets, f"{family} missing {expected}")

    def test_depth_ordering_is_monotonic(self):
        """Walking a flat stack front to back must never move backwards."""
        order = [bu.UNIVERSAL_EARLY, bu.UNIVERSAL_EARLY_MID, bu.UNIVERSAL_MID,
                 bu.UNIVERSAL_LATE_MID, bu.UNIVERSAL_LATE]
        keys = _flat_stack("blocks", 40)
        layout = bu.discover_layout(keys)
        seen = [order.index(bu.classify_universal_key(k, layout)) for k in keys]
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[0], 0)
        self.assertEqual(seen[-1], len(order) - 1)

    def test_dual_stack_spans_the_whole_axis(self):
        """FLUX double blocks sit early, single blocks run to the end."""
        layout = bu.discover_layout(FAMILIES["FLUX"])
        first = bu.classify_universal_key(
            "diffusion_model.double_blocks.0.img_attn.qkv.weight", layout)
        last = bu.classify_universal_key(
            "diffusion_model.single_blocks.37.linear1.weight", layout)
        self.assertEqual(first, bu.UNIVERSAL_EARLY)
        self.assertEqual(last, bu.UNIVERSAL_LATE)

    def test_unet_stages_land_in_sensible_buckets(self):
        layout = bu.discover_layout(FAMILIES["SDXL"])
        early = bu.classify_universal_key(
            "diffusion_model.input_blocks.0.1.transformer_blocks.0.attn1.to_q.weight", layout)
        late = bu.classify_universal_key(
            "diffusion_model.output_blocks.8.1.transformer_blocks.0.attn1.to_v.weight", layout)
        self.assertEqual(early, bu.UNIVERSAL_EARLY)
        self.assertEqual(late, bu.UNIVERSAL_LATE)

    def test_embedding_and_head_tokens(self):
        layout = bu.discover_layout(FAMILIES["FLUX"])
        inputs = ["diffusion_model.img_in.weight", "diffusion_model.txt_in.weight",
                  "diffusion_model.time_in.in_layer.weight",
                  "diffusion_model.x_embedder.proj.weight",
                  "diffusion_model.patch_embedding.weight",
                  "diffusion_model.caption_projection.linear_1.weight",
                  "diffusion_model.label_emb.0.0.weight"]
        heads = ["diffusion_model.final_layer.linear.weight",
                 "diffusion_model.proj_out.weight",
                 "diffusion_model.norm_out.linear.weight",
                 "diffusion_model.out.2.weight",
                 "diffusion_model.final_linear.weight"]
        for key in inputs:
            self.assertEqual(bu.classify_universal_key(key, layout),
                             bu.UNIVERSAL_INPUT, key)
        for key in heads:
            self.assertEqual(bu.classify_universal_key(key, layout),
                             bu.UNIVERSAL_OUTPUT, key)

    def test_unrecognised_key_falls_through(self):
        layout = bu.discover_layout(FAMILIES["FLUX"])
        self.assertEqual(
            bu.classify_universal_key("diffusion_model.mystery.weight", layout),
            bu.UNIVERSAL_OTHER)

    def test_classification_is_safe_with_an_empty_layout(self):
        layout = bu.discover_layout([])
        self.assertEqual(
            bu.classify_universal_key("diffusion_model.blocks.3.attn.weight", layout),
            bu.UNIVERSAL_OTHER)

    def test_index_beyond_discovered_size_is_clamped(self):
        layout = bu.discover_layout(_flat_stack("blocks", 10))
        result = bu.classify_universal_key("diffusion_model.blocks.99.attn.weight", layout)
        self.assertEqual(result, bu.UNIVERSAL_LATE)


class TestRealWorldRegressions(unittest.TestCase):
    """Cases found by running against real ComfyUI models, locked in here.

    Each of these silently landed in 'Other Tensors' before being fixed.
    """

    def test_sd3_negative_block_index(self):
        # ComfyUI's SD3 key map addresses the final block as "joint_blocks.-1".
        layout = bu.discover_layout(_flat_stack("joint_blocks", 24))
        key = "diffusion_model.joint_blocks.-1.context_block.adaLN_modulation.1.weight"
        self.assertEqual(bu._stack_match(key), ("joint_blocks", -1))
        self.assertEqual(bu.classify_universal_key(key, layout), bu.UNIVERSAL_LATE)

    def test_lumina_refiner_stacks_are_discovered(self):
        keys = (_flat_stack("noise_refiner", 2) + _flat_stack("context_refiner", 2)
                + _flat_stack("layers", 32))
        layout = bu.discover_layout(keys)
        self.assertEqual(layout.stacks,
                         {"noise_refiner": 2, "context_refiner": 2, "layers": 32})
        self.assertEqual(list(layout.stacks)[:2], ["noise_refiner", "context_refiner"])
        self.assertEqual(
            [k for k in keys if bu.classify_universal_key(k, layout) == bu.UNIVERSAL_OTHER],
            [])

    def test_auraflow_native_layer_stacks(self):
        # ComfyUI names these *_layers, not the diffusers *_transformer_blocks.
        keys = _flat_stack("double_layers", 4) + _flat_stack("single_layers", 32)
        layout = bu.discover_layout(keys)
        self.assertEqual(list(layout.stacks), ["double_layers", "single_layers"])

    def test_conditioning_embedders_are_input_not_other(self):
        layout = bu.discover_layout(_flat_stack("blocks", 28))
        for key in ("diffusion_model.ar_embedder.mlp.0.weight",      # PixArt
                    "diffusion_model.csize_embedder.mlp.0.weight",   # PixArt
                    "diffusion_model.t_block.1.weight",              # PixArt
                    "diffusion_model.adaln_single.linear.weight",    # LTX-Video
                    "diffusion_model.scale_shift_table",             # LTX-Video
                    "diffusion_model.txt_norm.weight",               # Qwen-Image
                    "diffusion_model.time_projection.1.weight",      # Wan
                    "diffusion_model.cond_seq_linear.weight",        # AuraFlow
                    "diffusion_model.positional_encoding",           # AuraFlow
                    "diffusion_model.init_x_linear.weight"):         # AuraFlow
            self.assertEqual(bu.classify_universal_key(key, layout),
                             bu.UNIVERSAL_INPUT, key)

    def test_head_token_does_not_swallow_attention_names(self):
        """A bare "_head_" substring also matched multi_head_attention."""
        layout = bu.discover_layout(_flat_stack("blocks", 28))
        self.assertEqual(bu.classify_universal_key(
            "diffusion_model.head.head.weight", layout), bu.UNIVERSAL_OUTPUT)
        self.assertEqual(bu.classify_universal_key(
            "diffusion_model.out.2.weight", layout), bu.UNIVERSAL_OUTPUT)
        for key in ("diffusion_model.multi_head_attention.weight",
                    "diffusion_model.some.head_dim_proj.weight"):
            self.assertNotEqual(bu.classify_universal_key(key, layout),
                                bu.UNIVERSAL_OUTPUT, key)

    def test_every_depth_bucket_populated_at_real_depths(self):
        order = (bu.UNIVERSAL_EARLY, bu.UNIVERSAL_EARLY_MID, bu.UNIVERSAL_MID,
                 bu.UNIVERSAL_LATE_MID, bu.UNIVERSAL_LATE)
        for depth in (12, 19, 24, 28, 40, 57, 60):
            keys = _flat_stack("blocks", depth)
            layout = bu.discover_layout(keys)
            counts = [sum(1 for k in keys
                          if bu.classify_universal_key(k, layout) == b) for b in order]
            self.assertEqual(sum(counts), depth, depth)
            self.assertTrue(all(c > 0 for c in counts), f"depth={depth}: {counts}")

    def test_last_block_reaches_the_final_bucket(self):
        """The top bucket boundary must include the deepest block."""
        for depth in (3, 5, 19, 60):
            keys = _flat_stack("blocks", depth)
            layout = bu.discover_layout(keys)
            self.assertEqual(bu.classify_universal_key(keys[-1], layout),
                             bu.UNIVERSAL_LATE, depth)

    def test_flux_controlnet_hint_input(self):
        self.assertEqual(
            bobs_blocks.classify_flux_key("diffusion_model.pos_embed_input.weight"),
            bobs_blocks.FLUX_IMAGE_HINT)


class TestUniversalPresets(unittest.TestCase):
    def test_registered_into_the_shared_tables(self):
        self.assertIn("UNIVERSAL", bobs_blocks.LORA_BLOCK_PRESETS)
        self.assertIs(bobs_blocks.ALL_BLOCKS["UNIVERSAL"], bu.ALL_UNIVERSAL_BLOCKS)
        self.assertEqual(bobs_blocks.TEXT_ENCODER_BLOCK["UNIVERSAL"],
                         bu.UNIVERSAL_TEXT_ENCODER)

    def test_every_preset_covers_every_block(self):
        for name, config in bu.UNIVERSAL_PRESETS.items():
            if name == "Custom":
                self.assertEqual(config, {})
                continue
            self.assertEqual(set(config["block_weights"]), set(bu.ALL_UNIVERSAL_BLOCKS), name)

    def test_resolve_block_strengths_handles_the_universal_family(self):
        result = bobs_blocks.resolve_block_strengths("UNIVERSAL", "Full (Normal LoRA)", 0.7, {})
        self.assertEqual(set(result), set(bu.ALL_UNIVERSAL_BLOCKS))
        for value in result.values():
            self.assertAlmostEqual(value, 0.7)

    def test_every_block_has_a_tooltip(self):
        for name in bu.ALL_UNIVERSAL_BLOCKS:
            text = bobs_blocks.tooltip_for(bu.UNIVERSAL_FAMILY, name)
            self.assertTrue(text and text != name, name)

    def test_shared_block_names_keep_per_family_tooltips(self):
        """FLUX/SDXL and UNIVERSAL both call a block "Text Encoder"."""
        shared = bu.UNIVERSAL_TEXT_ENCODER
        universal = bobs_blocks.tooltip_for(bu.UNIVERSAL_FAMILY, shared)
        flux = bobs_blocks.tooltip_for("FLUX", shared)
        self.assertNotEqual(universal, flux)
        self.assertIn("whatever the model uses", universal)
        self.assertIn("trigger words", flux)

    def test_registration_does_not_disturb_the_builtin_families(self):
        for family in bobs_blocks.BUILTIN_FAMILIES:
            self.assertIn(family, bobs_blocks.ALL_BLOCKS)
            self.assertIn(family, bobs_blocks.LORA_BLOCK_PRESETS)


if __name__ == "__main__":
    unittest.main()
