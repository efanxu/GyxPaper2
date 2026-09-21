# F DIRECTION SEMANTICS

## Frozen mapping

The names below are frozen from the executed query key value flow, not inferred from the word reverse.

| Direction | Query | Key and value | Updated branch | Disable field |
| --- | --- | --- | --- | --- |
| Macro to Fine | Fine history `[B*N,L,D]` | Macro Prompt `[B*N,P,D]` | Fine | `disable_macro_to_fine_cross` |
| Fine to Coarse | Coarse history `[B*N,L,D]` | Recent Fine history `[B*N,R,D]` | Coarse | `disable_reverse_cross` |

F4 sets `disable_reverse_cross=true`, so only Macro to Fine remains. F5 sets
`disable_macro_to_fine_cross=true`, so only Fine to Coarse remains. The Fine memory range is
`[L-min(cross_fusion_recent_len,L), L)`. Macro to Fine uses historical representations only and never
reads a future target or prediction horizon. These attention weights describe internal historical
representation association and are not causal horizon attributions.
