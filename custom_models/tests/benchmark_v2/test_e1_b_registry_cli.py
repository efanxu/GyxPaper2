import contextlib
import io
import sys
import unittest
from benchmark_v2.cli import main
from benchmark_v2.registry import load_registry

class E1BRegistryTests(unittest.TestCase):

    def test_registry_final_counts_and_e1_a_entries_unchanged(self):
        registry = load_registry()
        entries = registry.list()
        available = [entry for entry in entries if str(entry.runtime_status).startswith('AVAILABLE_')]
        self.assertEqual(len(entries), 28)
        self.assertEqual(len(available), 28)
        self.assertEqual(len(entries) - len(available), 0)
        self.assertEqual(sum((entry.supports_train for entry in entries)), 26)
        self.assertEqual(sum((entry.supports_non_trainable for entry in entries)), 2)
        self.assertEqual(registry.get('persistence').runtime_status, 'AVAILABLE_NON_TRAINABLE')
        self.assertEqual(registry.get('moving_average').runtime_status, 'AVAILABLE_NON_TRAINABLE')
        self.assertEqual(registry.get('gru').runtime_status, 'AVAILABLE_TRAINABLE')
        for model_id in ('dlinear', 'lightts'):
            entry = registry.get(model_id)
            self.assertEqual(entry.runtime_status, 'AVAILABLE_TRAINABLE')
            self.assertTrue(entry.supports_train)
            self.assertEqual(entry.adapter_kind, 'node_shared')
            self.assertFalse(entry.requires_graph)
        for model_id in ('tide', 'segrnn'):
            entry = registry.get(model_id)
            self.assertEqual(entry.runtime_status, 'AVAILABLE_TRAINABLE_HARDWARE_PREFLIGHT_REQUIRED')
            self.assertTrue(entry.supports_train)
            self.assertTrue(entry.supports_evaluate)
            self.assertTrue(entry.formal_hardware_preflight_required)
            self.assertEqual(entry.local_full_shape_status, 'FAIL_OOM')
            self.assertEqual(entry.non_oom_engineering_checks, 'PASS')
            self.assertEqual(entry.historical_local_full_shape_status, 'FAIL_OOM')
            self.assertIn('OutOfMemoryError', entry.historical_local_oom_error)
        self.assertEqual(registry.get('stcn_stgcn_unresolved').audit_status, 'IMPLEMENTED_CLEAN_ROOM')
        self.assertTrue(registry.get('transformer').supports_train)

    def test_registry_list_show_and_protocol_check_do_not_import_tslib(self):
        for name in tuple(sys.modules):
            if name.startswith('_benchmark_v2_tslib_'):
                sys.modules.pop(name)
        for argv in (['registry-list'], ['registry-show', '--model', 'dlinear'], ['protocol-check']):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(main(argv), 0)
            self.assertFalse(any((name.startswith('_benchmark_v2_tslib_') for name in sys.modules)))

    def test_e3_c_models_require_explicit_formal_run_id(self):
        for model_id in ('graph_wavenet', 'mtgnn'):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(['train', '--model', model_id])
            self.assertEqual(code, 2)
            self.assertIn('--run-id is required', stream.getvalue())
if __name__ == '__main__':
    unittest.main()
