# E3-C adaptive graph and identity fairness audit

## Result

PASS. The four models are intentionally different in how they encode node
identity, but all receive the same historical data, canonical node order,
target, loss, splits, normalization, seed, and graph-context identity.

- Graph WaveNet is the only model that uses both physical random-walk supports
  and an adaptive support.
- MTGNN and AGCRN validate the frozen graph context but use only learned graph
  structure in forward.
- STID validates node/graph context for dataset identity, then uses node and
  historical time embeddings with no graph propagation.
- No physical matrix was silently added to MTGNN, AGCRN, or STID.
- No graph propagation was silently added to STID.
- Learned parameters are checkpointed and optimizer-visible; frozen physical
  matrices are non-trainable and non-persistent.

## Leakage and isolation

Tests established invariance to target/mask changes, rejection of future
observed and future calendar inputs, canonical node-order failure on all
identity mismatches, batch-companion permutation/single-sample equivalence, and
strict reload identity reproduction. STID additionally passed cross-node
isolation and historical-anchor timestamp range/change tests.

## Selection firewall

Ordinary, exact full-shape, and limited real-data smoke results are engineering
path evidence only. They did not select hyperparameters, reduce capacity, rank
models, or support scientific conclusions. No test-set feedback entered
training or checkpoint selection.
