# E3-B DCRNN diffusion and decoder trace

Every DCGRU reset, update, and candidate transformation receives concatenated
current input and prior hidden state and uses the same frozen dual supports.
The ordered diffusion basis is:

`[X, P_forward X, P_forward^2 X, P_reverse X, P_reverse^2 X]`.

Identity appears exactly once. Multiplication is `support @ node_features`.
Encoder and decoder each have two independently parameterized DCGRU layers with
hidden size 64 and diffusion step 2.

For B=1:

- input: `(1,144,134,16)`;
- final encoder state: `(2,1,134,64)`;
- zero GO token: `(1,134,1)`;
- decoder: 10 autoregressive steps;
- each step output: `(1,134,1)`;
- stacked output: `(1,134,10)`.

`teacher_forcing=false`, `scheduled_sampling=false`, and
`curriculum_learning=false`. Steps 2...10 consume only the previous normalized
Patv_raw model prediction. Focused tests verify forward/reverse direction,
first/second order terms, reset/update/candidate gradients, both encoder and
decoder layers, output projection, and different output under dual random walk
versus identity-only diffusion.
