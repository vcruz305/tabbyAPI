"""Public aliases must identify the loaded model in both completion APIs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from endpoints.OAI.utils import common_, completion, chat_completion
from tests.test_timings import generation


class ResponseModelNameTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.loaded = self.root / "actual-3.05bpw"
        self.loaded.mkdir()
        (self.root / "public-qwen").symlink_to(self.loaded, target_is_directory=True)
        self.config = SimpleNamespace(
            model=SimpleNamespace(
                model_dir=str(self.root),
                use_dummy_models=False,
                dummy_model_names=["compatibility-model"],
            )
        )
        self.config_patch = patch.object(common_, "config", self.config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def test_canonical_and_verified_symlink_alias_are_preserved(self):
        for requested in ["actual-3.05bpw", "public-qwen", str(self.loaded)]:
            with self.subTest(requested=requested):
                self.assertEqual(common_.response_model_name(requested, self.loaded), requested)

    def test_unknown_names_cannot_be_echoed_as_served_models(self):
        other = self.root / "other-model"
        other.mkdir()
        (self.root / "another-public").symlink_to(other, target_is_directory=True)
        for requested in [None, "", "does-not-exist", "other-model", "another-public"]:
            with self.subTest(requested=requested):
                self.assertEqual(
                    common_.response_model_name(requested, self.loaded), "actual-3.05bpw"
                )

    def test_dummy_alias_requires_explicit_configuration(self):
        requested = "compatibility-model"
        self.assertEqual(common_.response_model_name(requested, self.loaded), "actual-3.05bpw")
        self.config.model.use_dummy_models = True
        self.assertEqual(common_.response_model_name(requested, self.loaded), requested)
        self.assertEqual(
            common_.response_model_name("not-configured", self.loaded), "actual-3.05bpw"
        )

    def test_cyclic_symlink_falls_back_to_loaded_name(self):
        (self.root / "cycle").symlink_to("cycle")
        self.assertEqual(common_.response_model_name("cycle", self.loaded), "actual-3.05bpw")

    def test_nested_quant_paths_do_not_match_other_models_same_basename(self):
        loaded = self.root / "qwen" / "exl3" / "4.05"
        other = self.root / "other" / "exl3" / "4.05"
        loaded.mkdir(parents=True)
        other.mkdir(parents=True)
        self.assertEqual(common_.response_model_name("qwen/exl3/4.05", loaded), "qwen/exl3/4.05")
        self.assertEqual(common_.response_model_name("other/exl3/4.05", loaded), "4.05")

    def test_stream_usage_and_nonstream_use_standard_model_field(self):
        for module in [completion, chat_completion]:
            with self.subTest(api=module.__name__):
                data = generation(delta_content="hi")
                alias = common_.response_model_name("public-qwen", self.loaded)
                chunk, payload, _, _ = module._compose_serialize_stream_chunk("req", data, alias)
                self.assertEqual(payload["model"], alias)
                self.assertEqual(json.loads(chunk)["model"], alias)
                self.assertNotIn("model_name", payload)
                usage, payload = module._compose_serialize_stream_usage_chunk(
                    "req", common_.get_usage_stats(data), 0, "stop", alias
                )
                self.assertEqual(json.loads(usage)["model"], alias)
                self.assertNotIn("model_name", payload)
                full = module._compose_response("req", [data], alias, True)
                self.assertEqual(full.model, alias)


if __name__ == "__main__":
    unittest.main()
