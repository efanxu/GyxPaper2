# E3-B implementation report

E3-B integrated GCN, classic STGCN, and DCRNN through a unified fail-closed
Native Graph Adapter and the frozen E3-A GraphBundle. All three are clean-room
PyTorch implementations and preserve their declared model identities.

Final engineering outcome:

- GCN: 6,763 parameters; ordinary/real/exact full-shape PASS; exact GTX 1060
  peak allocated 520,750,080 B and peak reserved 603,979,776 B.
- STGCN: 1,169,002 parameters; ordinary/real/exact full-shape PASS; exact peak
  allocated 2,541,089,792 B and peak reserved 2,791,309,312 B.
- DCRNN: 385,793 parameters; ordinary/real/exact full-shape PASS; exact peak
  allocated 6,511,126,016 B and peak reserved 6,538,919,936 B.

Every exact attempt used `B=32,T=144,N=134,C=16,H=10`, AMP, unchanged frozen
configuration, forward, masked MSE, and backward in an independent process.

Verification:

- E3-B focused: 17/17 PASS;
- full benchmark_v2: 138/138 PASS;
- hardware launcher regression: 9/9 PASS;
- compileall, Protocol check, and Graph Protocol check: PASS;
- ordinary smoke: 3/3 PASS;
- exact full-shape: 3/3 PASS;
- limited real SDWPF: 3/3 PASS;
- no final or historical E3-B FAIL attempt occurred.

Registry remains 28 entries: 22 trainable (12 locally exact PASS and 10
preflight-required), 2 non-trainable available, and 4 blocked/unavailable. The
TSLib allowlist remains exactly 18.

Formal training, formal evaluation, target high-memory preflight, and E3-C were
not started. The formal E3-B root was not created.
