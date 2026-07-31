import hashlib
import sys
import unittest
from pathlib import Path

from benchmark_v2.errors import ContractError
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream.tslib_loader import (
    ALLOWED_TSLIB_MODELS,
    PROJECT_ROOT,
    load_tslib_model_class,
    resolve_tslib_source,
)


EXPECTED_HASHES = {
    "dlinear": "e03acaf97e70c85ea6bda6776b3a62550f84470afa7bc53dffccb6121626e7eb",
    "lightts": "1de1583c4ee0bbfa0764fa9b158f09ac8bcd7ddf7f2d9f1b58ab1a03a7d13989",
    "tide": "4ab07dec4ae85f8b7c3062ff7d2fec00be342968d41cf09e48d599d8e40f6143",
    "segrnn": "764d4ad950ee65162c70746248754942cfb4ba661162d515caab0a4eedd12252",
    "transformer": "46ded6c516fd03951bce0a82c3f4245f0247efd01fb5bde775145cfbb6d8435d",
    "patchtst": "29835d4fedddd3cbee26cc60d6a54c04513f5a9ca32e4a0d08de2c9e059084e5",
    "itransformer": "7fdc721d041b0f8f63be8fa794ecd68422fd958c7c8d449026320fd9f368788e",
    "timexer": "b334d7869544d0a5de7a35501d0342d7bd0be4ebda06ccc045855a42819728c0",
    "timesnet": "f64c4bed1fd7347090044a0163bd4209c9f9ec1c1b19ceff47842df36b64bba7",
    "micn": "a0cb59254e850bfd1d94114a73126c290e6baf2ea700132ede8aade680bbedbf",
    "wpmixer": "5b549f5454864d19a230747f0060dc00732fb0de6b50a359173e0d7a6b04ce8f",
    "multipatchformer": "e8603a9d8b1e796a822171399a057bd3945cf51b38fb1661d91fe60ad79313b3",
    "timemixer": "bbf378cab03d16d3e7ae907f7b3cefbac89ae1504e036270578d5d496ffbd133",
    "tsmixer": "a82942ddc22cba59f4161f3ea480eb0e474eba6f0ab2d13f2bd949f100b2ce87",
    "frets": "2b6c9e0cd3d4f42bc74736343147812db94f029f1a8b13b098b9841705de5c18",
    "crossformer": "f5892f70eb3fc320f69fa0a9b2019b68537cc5a5e5aa2ec544c15127b19e36f0",
    "msgnet": "e6f8f3fbd4e0bc9d77c78e2409969c4fe4b2fc6560e7c1f5cced2a809d8e9e59",
    "timefilter": "6cbce92f112a9907fe6117ea23527bf1864c0ea00721a812d786e86e0a1b0724",
}


class TSLibLoaderTests(unittest.TestCase):
    def test_allowlist_is_exact_and_arbitrary_strings_fail_closed(self):
        self.assertEqual(
            ALLOWED_TSLIB_MODELS,
            (
                "dlinear", "lightts", "tide", "segrnn",
                "transformer", "patchtst", "itransformer", "timexer",
                "timesnet", "micn", "wpmixer", "multipatchformer",
                "timemixer", "tsmixer", "frets",
                "crossformer", "msgnet", "timefilter",
            ),
        )
        for name in ("../DLinear", "models.DLinear", ""):
            with self.assertRaisesRegex(ContractError, "not allowlisted"):
                resolve_tslib_source(name)

    def test_resolved_paths_and_hashes_are_exact(self):
        for model_id, expected_hash in EXPECTED_HASHES.items():
            source = resolve_tslib_source(model_id)
            path = Path(source.source_path)
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent, PROJECT_ROOT / "Time-Series-Library/models")
            self.assertEqual(source.source_sha256, expected_hash)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash
            )
            self.assertEqual(source.upstream_project, "THUML/Time-Series-Library")
            self.assertFalse(source.source_modified)

    def test_registry_listing_does_not_lazy_import_tslib_sources(self):
        before = set(sys.modules)
        registry = load_registry()
        registry.list()
        new_modules = set(sys.modules) - before
        self.assertFalse(
            any(name.startswith("_benchmark_v2_tslib_") for name in new_modules)
        )
        self.assertFalse(
            any(name.startswith("benchmark_v2.models.") for name in new_modules)
        )

    def test_create_loads_only_the_explicit_source_and_restores_sys_path(self):
        original = list(sys.path)
        for model_id in ALLOWED_TSLIB_MODELS:
            model_class, source = load_tslib_model_class(model_id)
            self.assertEqual(model_class.__name__, "Model")
            self.assertEqual(
                Path(sys.modules[model_class.__module__].__file__).resolve(),
                Path(source.source_path),
            )
            self.assertEqual(sys.path, original)


if __name__ == "__main__":
    unittest.main()
