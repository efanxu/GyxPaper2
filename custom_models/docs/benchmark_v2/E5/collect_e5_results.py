from __future__ import annotations

import argparse
from benchmark_v2.experiments.e5_common_loss.aggregation import aggregate

parser = argparse.ArgumentParser()
parser.add_argument("--output-root")
parser.add_argument("--require-complete", action="store_true")
args = parser.parse_args()
print(aggregate(output_root=args.output_root, require_complete=args.require_complete))
