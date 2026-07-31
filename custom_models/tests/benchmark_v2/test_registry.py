import sys
import unittest

from benchmark_v2.errors import ModelUnavailableError
from benchmark_v2.registry import load_registry


class RegistryTests(unittest.TestCase):
    def test_explicit_registry_is_complete(self):
        registry = load_registry()
        self.assertEqual(len(registry.list()), 28)
        self.assertEqual(len({e.canonical_id for e in registry.list()}), 28)
        aliases = [a.casefold() for e in registry.list() for a in e.aliases]
        self.assertEqual(len(aliases), len(set(aliases)))
        self.assertNotIn("stcn", aliases)
        self.assertNotIn("stgcn", aliases)

    def test_registry_list_does_not_import_old_runtimes(self):
        before = set(sys.modules)
        load_registry().list()
        newly_imported = set(sys.modules) - before
        self.assertNotIn("st_mgprompt", newly_imported)
        self.assertFalse(any(name.startswith("Time-Series-Library") for name in newly_imported))

    def test_e3_c_graph_wavenet_is_available(self):
        entry = load_registry().get("graph_wavenet")
        self.assertEqual(entry.runtime_status, "AVAILABLE_TRAINABLE")
        self.assertTrue(entry.supports_train)
        self.assertTrue(entry.supports_evaluate)

    def test_known_errors_and_unresolved_identity(self):
        registry = load_registry()
        self.assertEqual(
            registry.get("stcn_stgcn_unresolved").audit_status,
            "IMPLEMENTED_CLEAN_ROOM",
        )
        self.assertIsNone(registry.get("stcn_stgcn_unresolved").values.get("recommended_canonical_name"))
        self.assertIn("tensor dimension 0 mismatch", registry.get("segrnn").known_smoke_error)
        self.assertIn("154", registry.get("micn").known_smoke_error)
        self.assertIn("58", registry.get("micn").known_smoke_error)
        self.assertIn("18", registry.get("multipatchformer").known_smoke_error)
        self.assertIn("19", registry.get("multipatchformer").known_smoke_error)
        self.assertIn("season_list[1]", registry.get("timemixer").known_smoke_error)


if __name__ == "__main__":
    unittest.main()
