from __future__ import annotations
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmark_v2.graph import MATRIX_NAMES, load_graph_bundle

class GraphProtocolTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_graph_bundle(validate_location_source=False)

    def test_explicit_structure_and_numeric_validity(self) -> None:
        self.assertEqual(self.bundle.spec.node_count, 134)
        self.assertEqual(len(self.bundle.ordered_node_ids), 134)
        self.assertEqual(set(self.bundle.matrices), set(MATRIX_NAMES))
        for matrix in self.bundle.matrices.values():
            self.assertEqual(matrix.shape, (134, 134))
            self.assertTrue(np.isfinite(matrix).all())

    def test_location_line_endings_do_not_change_acceptance(self) -> None:
        rows = [[node_id, float(index), float(index + 1), 0.0] for index, node_id in enumerate(self.bundle.ordered_node_ids)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, ending in (('lf.csv', '\n'), ('crlf.csv', '\r\n')):
                path = root / name
                with path.open('w', encoding='utf-8', newline='') as handle:
                    writer = csv.writer(handle, lineterminator=ending)
                    writer.writerow(['TurbID', 'x', 'y', 'Ele'])
                    writer.writerows(rows)
                loaded = load_graph_bundle(location_path=path)
                self.assertEqual(loaded.ordered_node_ids, self.bundle.ordered_node_ids)
if __name__ == '__main__':
    unittest.main()
