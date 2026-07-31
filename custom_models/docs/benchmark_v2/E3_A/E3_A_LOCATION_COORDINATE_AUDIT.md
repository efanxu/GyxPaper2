# E3-A location and coordinate audit

Source: `dataset/sdwpf_turb_location_elevation.csv`

SHA256: `233a120f4db2b52402bf4dc7c90631ac9301d6af236a56c6a245e46beec1f236`

Columns and pandas types are `TurbID:int64`, `x:float64`, `y:float64`, and `Ele:float64`. Ranges are:

- x: 0.0 to 5501.4529
- y: 0.0 to 12121.00426
- Ele: 1391.2000732421875 to 1469.800048828125

Local code establishes Cartesian semantics: `dataset/clean_sdwpf_data.py::build_knn_neighbors` computes `sqrt(sum((coord_i-coord_j)^2))` over x/y, and `st_mgprompt/graph_prior.py::build_distance_prior_graph` independently uses squared Euclidean x/y. There is no latitude/longitude field or haversine path.

The local file and documentation do not declare the x/y or Ele unit. Units are therefore frozen honestly as `SOURCE_UNIT_UNSPECIFIED`, not guessed as metres. This does not change the distance decision because the project already treats x/y as one Cartesian coordinate system.

`Ele` is retained in node metadata and is not used in base edge distance. This avoids unverified horizontal/elevation scale mixing, avoids importing ST-MGPrompt private design, and preserves a neutral reproducible horizontal KNN graph.
