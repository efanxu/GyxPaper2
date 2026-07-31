# E3-C model specifications

## Graph WaveNet

The model applies a gated dilated causal temporal stack. Every layer combines
residual and skip paths with diffusion graph convolution over three supports:
frozen `P_forward`, frozen `P_reverse`, and
`softmax(ReLU(E1 @ E2))`. Diffusion terms are order 1 and 2 for every support.
The final valid causal position is projected directly to 10 horizons.

## MTGNN

The graph constructor uses two learned 40-dimensional node embeddings,
antisymmetric pair scores, `tanh(3·score)`, and a deterministic directed row
top-k mask with exactly 20 non-self edges. Three dilated inception layers use
kernels 2/3/6/7 and align to the shortest branch. Each layer performs forward
and reverse MixProp to depth 2 with `propalpha=0.05`. Frozen physical matrices
are validated as graph context but never enter forward.

## AGCRN

DAGG is `softmax(ReLU(E @ Eᵀ))`. Every adaptive graph convolution uses exactly
two bases, `T0=I` and `T1=A`. NAPL generates node-specific weights and biases
from the same learned `(134,10)` embeddings. Two recurrent layers of width 64
consume all 144 historical steps; a direct linear head emits 10 horizons.
No predefined physical adjacency enters forward.

## STID

Each node's full `(144,16)` history is projected to a 32-dimensional series
embedding and concatenated with learned node, time-of-day, and day-of-week
embeddings, each width 32. Three residual MLP blocks process the resulting
128-dimensional identity representation before a direct 10-horizon head.
There is no graph construction or cross-node propagation.

Parameter counts from ordinary smoke are respectively 307,362; 4,686,714;
804,550; and 183,242. Every trainable branch named above received gradients in
focused tests.
