# E3-A node order audit

`TurbID` is `int64` in both parquet schemas and is canonically serialized as a JSON integer. `SDWPFDataProvider.from_files` defines the actual N-axis as:

`sorted(model_df["TurbID"].astype(int).unique().tolist())`

The resulting order is integer `1..134`. This conclusion comes from the provider code and actual file audit; it is not inferred from the location CSV row order.

Input and target contain 7,042,906 aligned `(Tmstamp,TurbID)` keys, no duplicate key, no null node ID, and the same 134-node set. The location file contains the same set, no duplicate node ID, no null coordinate, and no duplicate `(x,y)`.

Frozen node-order record:

`<removed-content-record>`

Full evidence is in `node_order_audit.json` and `protocol/graph_v1/node_order_v1.json`. Input/target node-source records cover node schema only, deliberately excluding feature, target, and mask values.
