# E5 non-trainable reference policy

Persistence and MovingAverage are evaluate-only references. They create no
optimizer, checkpoint, best epoch, or checkpoint selection. Their artifact
records `training_loss=NOT_APPLICABLE`, `trained_with_common_loss=false`,
`common_loss_evaluation_applied=true`, and computes the common loss only as a
normalized-space diagnostic alongside official H3/H6/H10 metrics.
