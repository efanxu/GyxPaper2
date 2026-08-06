# E3-A graph protocol decision

No qualifying common protocol existed, so the deterministic fallback was applied.

- graph ID: `sdwpf_physical_knn_v1`
- coordinate system: Cartesian x/y
- distance: two-dimensional Euclidean
- elevation: metadata only
- k rule: first k in 1..20 whose maximum-symmetrized directed KNN support is connected
- exact-distance tie: canonical node-order index
- component counts at k=1,2,3,4: 48, 7, 2, 1
- selected k: 4
- directed edges: 536
- undirected edges: 298
- sigma: 921.5755719183301 source-coordinate units
- weight: `exp(-(distance/sigma)^2)`
- base self-loop: none
- symmetrization: `maximum(A_directed, A_directed.T)`

All calculations use CPU float64. Identity matrices are C-contiguous little-endian float64, finite, rounded to 12 decimals, and have negative zero normalized to zero. `.npy` container bytes are not identity-bearing.

`lambda_max=2.0` is the theoretical normalized-Laplacian upper bound, not a numerically solved or performance-selected value. Thus `L_tilde=L_sym-I`.

Changing feature, target, or mask values cannot change graph identity because edge construction accepts only canonical node IDs, x/y, and protocol constants. A location file byte change changes location identity and graph-protocol identity. If only unused elevation changes, node metadata and protocol identity change, while base matrix records and graph-bundle content record remain unchanged.
