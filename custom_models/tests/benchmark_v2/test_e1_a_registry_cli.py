import contextlib
import io
import sys
import unittest
from benchmark_v2.cli import main
from benchmark_v2.registry import load_registry

class E1ARegistryAndCLITests(unittest.TestCase):

    def test_registry_counts_factories_and_lazy_import(self):
        for name in ['benchmark_v2.models.persistence', 'benchmark_v2.models.moving_average', 'benchmark_v2.models.gru']:
            sys.modules.pop(name, None)
        registry = load_registry()
        entries = registry.list()
        available = [entry for entry in entries if str(entry.runtime_status).startswith('AVAILABLE_')]
        self.assertEqual(len(entries), 28)
        self.assertEqual(len(available), 28)
        self.assertEqual(len(entries) - len(available), 0)
        self.assertEqual(sum((entry.supports_train for entry in entries)), 26)
        self.assertEqual(sum((entry.supports_non_trainable for entry in entries)), 2)
        self.assertNotIn('benchmark_v2.models.gru', sys.modules)
        protocol = __import__('benchmark_v2.protocol', fromlist=['load_protocol']).load_protocol()
        gru = registry.create_model('gru', config={}, protocol=protocol)
        self.assertEqual(type(gru).__name__, 'NodeSharedGRU')
        for model_id in ('persistence', 'moving_average', 'gru'):
            entry = registry.get(model_id)
            self.assertTrue(entry.factory)
            self.assertTrue(entry.implementation_module)
        self.assertEqual(registry.get('dlinear').runtime_status, 'AVAILABLE_TRAINABLE')
        self.assertEqual(registry.get('stcn_stgcn_unresolved').audit_status, 'IMPLEMENTED_CLEAN_ROOM')

    def test_train_non_trainable_and_unavailable_fail_closed(self):
        for model_id in ('persistence', 'moving_average'):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(['train', '--model', model_id])
            self.assertEqual(code, 2)
            self.assertIn('unavailable for training', stream.getvalue())
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(['train', '--model', 'graph_wavenet'])
        self.assertEqual(code, 2)
        self.assertIn('--run-id is required', stream.getvalue())

    def test_registry_show_does_not_read_data(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = main(['registry-show', '--model', 'persistence'])
        self.assertEqual(code, 0)
        self.assertIn('AVAILABLE_NON_TRAINABLE', stream.getvalue())
if __name__ == '__main__':
    unittest.main()
