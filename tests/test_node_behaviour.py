"""Node-level tests: the parts of bobs_lora_loader that need no real model.

comfy.* and folder_paths are stubbed just enough to import the module; the
logic under test (mismatch detection, report formatting) is ours.
"""

import importlib.util
import os
import sys
import types
import unittest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)


def _install_stubs():
    comfy = types.ModuleType("comfy")
    lora = types.ModuleType("comfy.lora")
    utils = types.ModuleType("comfy.utils")
    lora.model_lora_keys_unet = lambda model, key_map=None: key_map
    lora.model_lora_keys_clip = lambda model, key_map=None: key_map
    lora.load_lora = lambda sd, km, log_missing=True: {}
    utils.load_torch_file = lambda p, safe_load=False, **kw: {}
    comfy.lora, comfy.utils = lora, utils
    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.lora", lora)
    sys.modules.setdefault("comfy.utils", utils)
    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda t: []
    fp.get_full_path = lambda t, n: None
    sys.modules.setdefault("folder_paths", fp)


_install_stubs()

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


BB = _load("bobs_blocks")
BU = _load("bobs_universal")
NODE = _load("bobs_lora_loader")


def _group(counts):
    """Build a grouped-patches dict with the given per-block patch counts."""
    return {name: {f"{name}.{i}": object() for i in range(n)}
            for name, n in counts.items()}


class TestMismatchWarning(unittest.TestCase):
    def test_everything_in_the_catch_all_warns(self):
        grouped = _group({BB.FLUX_OTHER: 40})
        msg = NODE._mismatch_warning("FLUX", BB.ALL_FLUX_BLOCKS, grouped,
                                     BB.FLUX_TEXT_ENCODER, BB.FLUX_OTHER)
        self.assertIsNotNone(msg)
        self.assertIn("Other Tensors", msg)
        self.assertIn("Universal", msg)

    def test_everything_in_one_named_block_warns(self):
        grouped = _group({BB.FLUX_EARLY_DOWN: 40})
        msg = NODE._mismatch_warning("FLUX", BB.ALL_FLUX_BLOCKS, grouped,
                                     BB.FLUX_TEXT_ENCODER, BB.FLUX_OTHER)
        self.assertIsNotNone(msg)
        self.assertIn("Early Downsampling", msg)

    def test_near_total_pileup_warns(self):
        """Real SDXL keys through the FLUX classifier leave 98.7%, not 100%,
        in the catch-all: a few hit the shared proj_out token."""
        grouped = _group({BB.FLUX_OTHER: 1674, BB.FLUX_FINAL: 22})
        msg = NODE._mismatch_warning("FLUX", BB.ALL_FLUX_BLOCKS, grouped,
                                     BB.FLUX_TEXT_ENCODER, BB.FLUX_OTHER)
        self.assertIsNotNone(msg)
        self.assertIn("99%", msg)

    def test_a_healthy_distribution_does_not_warn(self):
        grouped = _group({BB.FLUX_EARLY_DOWN: 16, BB.FLUX_MID_DOWN: 16,
                          BB.FLUX_CORE: 60, BB.FLUX_LATE_UP: 18})
        self.assertIsNone(
            NODE._mismatch_warning("FLUX", BB.ALL_FLUX_BLOCKS, grouped,
                                   BB.FLUX_TEXT_ENCODER, BB.FLUX_OTHER))

    def test_text_encoder_only_lora_does_not_warn(self):
        """A TE-only LoRA is legitimate, not a family mismatch."""
        grouped = _group({BB.FLUX_TEXT_ENCODER: 40})
        self.assertIsNone(
            NODE._mismatch_warning("FLUX", BB.ALL_FLUX_BLOCKS, grouped,
                                   BB.FLUX_TEXT_ENCODER, BB.FLUX_OTHER))

    def test_tiny_loras_are_not_judged(self):
        grouped = _group({BB.FLUX_OTHER: 3})
        self.assertIsNone(
            NODE._mismatch_warning("FLUX", BB.ALL_FLUX_BLOCKS, grouped,
                                   BB.FLUX_TEXT_ENCODER, BB.FLUX_OTHER))

    def test_each_node_declares_its_catch_all(self):
        self.assertEqual(NODE.BobsLoraLoaderFlux.CATCH_ALL_BLOCK, BB.FLUX_OTHER)
        self.assertEqual(NODE.BobsLoraLoaderSdxl.CATCH_ALL_BLOCK, BB.SDXL_OTHER)
        self.assertEqual(NODE.BobsLoraLoaderUniversal.CATCH_ALL_BLOCK, BU.UNIVERSAL_OTHER)
        for cls in (NODE.BobsLoraLoaderFlux, NODE.BobsLoraLoaderSdxl,
                    NODE.BobsLoraLoaderUniversal):
            self.assertIn(cls.CATCH_ALL_BLOCK, cls.BLOCKS)


class TestReport(unittest.TestCase):
    def test_block_names_fit_the_report_column(self):
        every = BB.ALL_FLUX_BLOCKS + BB.ALL_SDXL_BLOCKS + BU.ALL_UNIVERSAL_BLOCKS
        self.assertLessEqual(max(len(b) for b in every), 40)

    def test_report_lists_every_block_and_a_total(self):
        grouped = _group({BB.SDXL_INPUT_BLOCKS: 9})
        strengths = {b: 1.0 for b in BB.ALL_SDXL_BLOCKS}
        applied = {BB.SDXL_INPUT_BLOCKS: 9}
        report = NODE._format_report("SDXL", "x.safetensors", "Custom",
                                     BB.ALL_SDXL_BLOCKS, grouped, strengths,
                                     applied, "architecture: SDXL")
        for block in BB.ALL_SDXL_BLOCKS:
            self.assertIn(block, report)
        self.assertIn("architecture: SDXL", report)
        self.assertIn("TOTAL", report)


class TestNodeRegistration(unittest.TestCase):
    def test_three_nodes_registered_with_display_names(self):
        self.assertEqual(set(NODE.NODE_CLASS_MAPPINGS),
                         {"BobsLoraLoaderFlux", "BobsLoraLoaderSdxl",
                          "BobsLoraLoaderUniversal"})
        self.assertEqual(set(NODE.NODE_CLASS_MAPPINGS),
                         set(NODE.NODE_DISPLAY_NAME_MAPPINGS))

    def test_input_types_are_well_formed(self):
        for cls in NODE.NODE_CLASS_MAPPINGS.values():
            spec = cls.INPUT_TYPES()
            self.assertEqual(set(spec), {"required", "optional"}, cls.__name__)
            self.assertIn("clip", spec["optional"])
            for field in ("model", "lora_name", "strength", "preset"):
                self.assertIn(field, spec["required"], cls.__name__)
            for block in cls.BLOCKS:
                self.assertIn(block, spec["required"], f"{cls.__name__}/{block}")
                opts = spec["required"][block][1]
                self.assertTrue(opts["tooltip"], block)
                self.assertNotEqual(opts["tooltip"], block, block)

    def test_widget_order_places_blocks_after_the_fixed_widgets(self):
        for cls in NODE.NODE_CLASS_MAPPINGS.values():
            names = list(cls.INPUT_TYPES()["required"])
            self.assertEqual(names[:4], ["model", "lora_name", "strength", "preset"])
            self.assertEqual(names[4:], list(cls.BLOCKS), cls.__name__)

    def test_outputs_are_model_clip_info(self):
        for cls in NODE.NODE_CLASS_MAPPINGS.values():
            self.assertEqual(cls.RETURN_TYPES, ("MODEL", "CLIP", "STRING"))
            self.assertEqual(cls.RETURN_NAMES, ("MODEL", "CLIP", "info"))


if __name__ == "__main__":
    unittest.main()
