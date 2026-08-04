# E5 aggregation policy

`e5-aggregate --require-complete` fails closed unless 24 formal common-loss
train runs, 2 formal evaluate-only references, and the formal Batch4 A8
prerequisite reference are all complete and identity-matched with H3/H6/H10
metrics. On failure it writes
no final XLSX/CSV/Markdown. Smoke results are never eligible. Persistence and
MovingAverage are excluded from the trained-architecture ranking.
