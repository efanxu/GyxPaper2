# E2-D TimeFilter token and mask trace

`patch_len=16`, stride 16, `144/16=9` patches per variable, 16 variables, and
144 total tokens. Historical input is permuted to feature-major `(BN,16,144)`,
flattened to `(BN,2304)`, unfolded into 144 non-overlapping 16-step patches,
and projected to `(BN,144,512)`.

For token `k`, `N=9`:

- S selects other variables at the same temporal patch offset (`index % 9`);
- T selects other temporal patches in the same variable block
  (`index // 9`);
- ST selects all remaining cross-variable/cross-time tokens;
- the diagonal is removed from S/T/ST and added back as an identity mask.

The mask is `(144,3,144)`. Per block, learned adjacency is
`(BN,8,144,144)`. `alpha=0.1` controls KNN filtering and `top_p=0.5` controls
the three-region MoE router. Two graph-filter blocks, all graph projections,
GCNs, FFNs, gate weights, patch projection, and output head receive finite
gradients.

Training uses noisy routing; resetting the seed reproduces its result.
Evaluation disables noise and is exactly deterministic. E2-D preserves the
upstream hard top-p forward mask and supplies a straight-through gradient for
the gate. The backbone's MoE auxiliary loss is recorded as unused and is never
added to the frozen masked-MSE objective.
