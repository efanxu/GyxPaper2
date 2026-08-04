# E5 result schema

Table A contains 26 trainable benchmark runs plus A8 (27 rows). Table B
contains 24 trainable rows plus Persistence and MovingAverage (2 rows) and the
independent Batch4 A8 prerequisite reference. The complete active appendix
contains exactly 27 rows.
Each trainable row includes model/category/mode/loss identity, H3/H6/H10
Score/MAE/RMSE/R2, average Score, best epoch, parameter counts, training and
preflight status. Non-trainable rows explicitly use
`training_loss=NOT_APPLICABLE`.
