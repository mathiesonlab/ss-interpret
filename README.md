# ss-interpret

Code and figures for the analysis in [work]. 
Scripts in `scripts/` can be used to fully reproduce the analysis. 
Environment should be python 3.12 with tensorflow (see `requirements.txt`). Note
the comment about `pylibseq`.

## Data

The real/sim dataset (200000 windows per population, large filesize!) can be 
[found here](https://upenn.box.com/s/lh3ayj272l1si4mc85jx5fosjbp3dqif). 
It also includes predictions from models and learned features.
For each population:

- `X.npy` are raw matrices in shape (n, n_haps, n_snps, 2)
- `y.npy` are corresponding binary labesl (0: sim, 1: real) shape (n, )
- `computed` contains predictions in npz files. Each `_preds_lf.npz` file contains:
  - An array `preds` containing model logit outputs of shape (n,)
  - An array `lf` containing model last-layer outputs of shape (n, ll-size)
    - New models are size 64
    - pg-gan models are size 128

The trained CNNs can be 
[found here](https://upenn.box.com/s/ndf9pfnsjw4tknyrz3a0sizxxjz5ov0o).

The pg-gan data (simulator parameters, discriminators) can be 
[found here](https://upenn.box.com/s/i1sv1fxkfl9at3k5u9zjn4dgaotta8sn).

## Reproduction

Create an environment using `requirements.txt`.

Downloading and unzipping the above data and running scripts from `scripts/`
will reproduce all figures and figure data.

If you are reproduce CNN trainings, you may have to adjust filtered runs in
`utils.py`. 
