import sys
import unittest
from pathlib import Path
from benchmark_v2.errors import ContractError
from benchmark_v2.registry import load_registry
from benchmark_v2.upstream.tslib_loader import ALLOWED_TSLIB_MODELS, PROJECT_ROOT, load_tslib_model_class, resolve_tslib_source

class TSLibLoaderTests(unittest.TestCase):

    def test_allowlist_is_exact_and_arbitrary_strings_fail_closed(self):
        self.assertEqual(ALLOWED_TSLIB_MODELS, ('dlinear', 'lightts', 'tide', 'segrnn', 'transformer', 'patchtst', 'itransformer', 'timexer', 'timesnet', 'micn', 'wpmixer', 'multipatchformer', 'timemixer', 'tsmixer', 'frets', 'crossformer', 'msgnet', 'timefilter'))
        for name in ('../DLinear', 'models.DLinear', ''):
            with self.assertRaisesRegex(ContractError, 'not allowlisted'):
                resolve_tslib_source(name)

    def test_registry_listing_does_not_lazy_import_tslib_sources(self):
        before = set(sys.modules)
        registry = load_registry()
        registry.list()
        new_modules = set(sys.modules) - before
        self.assertFalse(any((name.startswith('_benchmark_v2_tslib_') for name in new_modules)))
        self.assertFalse(any((name.startswith('benchmark_v2.models.') for name in new_modules)))

    def test_create_loads_only_the_explicit_source_and_restores_sys_path(self):
        original = list(sys.path)
        for model_id in ALLOWED_TSLIB_MODELS:
            model_class, source = load_tslib_model_class(model_id)
            self.assertEqual(model_class.__name__, 'Model')
            self.assertEqual(Path(sys.modules[model_class.__module__].__file__).resolve(), Path(source.source_path))
            self.assertEqual(sys.path, original)
if __name__ == '__main__':
    unittest.main()
