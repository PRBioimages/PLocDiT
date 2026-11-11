# SUBdit — Subcellular Locating Diffusion Transformer

A self-contained PyTorch training repository that adapts the official DiT latent
diffusion transformer for **subcellular protein localization** on multi-channel
fluorescence microscopy images (CYPM / CYNP_ALL / MTLS / MTNP / NCNP).

Key extensions over the base DiT recipe:

| Feature | What it does |
| --- | --- |
| **Structured conditioning** | A frozen CNN or HPA-VAE struct encoder (2 channels) produces struct latents that are *sequence-concatenated* with the protein latents into a single 512-token DiT sequence |
| **REPA feature alignment** | A frozen MAE-CellS-ProtS-Pool teacher penalises the student (DiT layer 4) for drifting away from a subcellular-aware feature space via `1 - cosine` |
| **Latent adversarial** | A transition discriminator (`lddgan`, DDPM) or endpoint discriminator (`plocgan`, flow) pushes the posterior x_{t-1} to match the real one, with an auxiliary *quantitative-label head* for partial-label candidate selection |
| **DFA-CFG** | Frequency-aware classifier-free guidance (`v_u + w_low*(v_c^low-v_u^low) + v_u^high + w_high*(v_c^high-v_u^high)`) with FFT circular low-pass, `dual_linear` time schedule, and per-sample `time_schedule` |
| **Label CFG** | `apply_cfg_train_dropout` zeroes both `z_struct` and `y` with probability `p`, but keeps the *clean* labels available for the ratio regression and the discriminator label head |

Three losses are recommended for a full run:

```
--diffusion-type ddpm --dit-parameterization x0 \
--repa-coef 0.3 --latent-adv-coef 0.1 --latent-adv-aux-coef 0.1 \
--ratio-label-coef 0.1 --cfg-train-dprob 0.15
```

---

## Repository layout

```
SUBdit/
├── train.py, sample.py, sample_test.py, sample_cfg.py
├── core/
│   ├── models.py                DiTWithStructSeq (seq_concat) + REPAProjector + DFACFGWrapper
│   ├── dit_model_build.py       Build helpers mirroring train.py defaults
│   ├── struct_encoder.py        CNN struct encoder
│   ├── struct_hpa_encoder.py    Frozen HPA VAE encoder/decoder (main AE + struct AE)
│   ├── subcell_teacher.py       SubCell REPA teacher loader (strict, head-only missing allowed)
│   ├── repa.py                  REPA loss + cosine
│   ├── latent_discriminator.py  LDDGAN / PLOC-GAN transition discriminators + hinge losses
│   ├── freq_loss.py             High-/low-band FFT MSE (circular low-pass, DC @ h//2)
│   ├── fid_inception_weights.py InceptionV3 FID helper
│   └── download.py
├── diffusion/                   GaussianDiffusion + FlowMatching (with DFA-CFG hooks)
├── utils/
│   └── dynamic_freq_sampling.py DFACFGConfig + DFACFGWrapper (time_schedule switch fixed)
├── pretrained/                  hpa_pretrained.bin + all_channels_MAE-CellS-ProtS-Pool.pth
├── env/                         environment.yml
├── scripts/                     eval_*, vis_*, generate_*, finetune_*, run_*.sh ...
├── requirements.txt
└── LICENSE.txt
```

---

## Setup

```bash
conda env create -f env/environment.yml
conda activate yangx       # or whatever you named it
pip install -r requirements.txt
```

The HPA VAE weights and the SubCell REPA teacher are expected at:

```
pretrained/hpa_pretrained.bin                          (164 MB)
pretrained/all_channels_MAE-CellS-ProtS-Pool.pth       (333 MB)
```

Override locations via `--main-ae-ckpt`, `--hpa-vae-ckpt`, and `--subcell-model-path`.

---

## Dataset layout

The default convention is the one used by CYPM / CYNP_ALL / MTLS / MTNP / NCNP:

```
<data-root>/
  train/    BGYImg*.png   (3-channel BGR = nucleus / protein / microtubule)
  val/      BGYImg*.png
  test/     BGYImg*.png
  trainlabels_CyPM.h5     (key "labels", shape N×2 float32)
  vallabels_CyPM.h5
  testlabels_CyPM.h5
```

The labels are 2-D floats `(p_cytoplasm, p_nucleus)` in [0, 1].

The protein channel (`ch1`, plane G) goes through the HPA VAE → 4×64×64 protein latent.
The struct channels (default `0 2`) go through the struct encoder → 8×64×64 struct latent,
concatenated into a 512-token sequence of width 768 for DiT-B/4.

---

## Training

Minimal example (CYNP_ALL, 3 losses, x0 parameterization, DDPM):

```bash
python train.py \
  --data-root /path/to/CYNP_ALL --dataset-h5-tag CyNp \
  --model DiT-B/4 --struct-backend hpa_vae --struct-ch-indices 0 2 \
  --dit-parameterization x0 --diffusion-type ddpm \
  --repa-coef 0.3 --latent-adv-coef 0.1 --latent-adv-aux-coef 0.1 \
  --ratio-label-coef 0.1 --cfg-train-dprob 0.15 \
  --epochs 25 --batch-size 16 --results-dir results_smoke
```

Environment shortcuts:

- `SUBDIT_DATA_ROOT` — fallback data root when `--data-root` is empty (overrides the hard-coded default)

The training script:

- probes the initial loss on one batch before stepping (`init-loss`)
- logs a running-average train loss every `--log-every` steps
- saves checkpoints every `--ckpt-every` steps (defaults to epoch end)
- optionally enables Exponential Moving Average via `--ema`

Resume training with:

```bash
python train.py --resume results_smoke/checkpoints/0005000.pt --results-dir results_smoke
```

Struct-encoder missing keys in a resume checkpoint raise (no silent partial init).

---

## Sampling

Plain DDIM / DDPM:

```bash
python sample.py \
  --dit-ckpt results_smoke/checkpoints/epoch_0025.pt \
  --struct-img-dir /path/to/CYNP_ALL/test --struct-ch-indices 0 2 \
  --num-sampling-steps 250 --sampler ddim --ddim-eta 0.0 \
  --output-dir gen_samples
```

The script reads `scale_factor` from the checkpoint and **divides the latent by it**
before `ae.decode`, matching the training-side convention. The struct backend is
inferred from `train_args.struct_backend` inside the checkpoint.

DFA-CFG comparison (no-CFG vs DFA-CFG side-by-side):

```bash
python sample_cfg.py \
  --dit-ckpt results_smoke/checkpoints/epoch_0025.pt \
  --img-dir /path/to/CYNP_ALL/test --labels-h5 /path/to/testlabels_CyNp.h5 \
  --cfg-scale 1.25 \
  --dfa-schedule dual_linear --dfa-ratio 0.3 \
  --ddim-steps 150 --n-samples 16 --out-dir cfg_compare
```

`--no-dfa-cfg` falls back to plain `CFGModelWrapper` (no latent-channel mixing, no
DFACFGWrapper → avoids the old bug where disabling DFA-CFG silently dropped the
guidance).

---

## Evaluation

```bash
python scripts/eval_dit_metrics.py --help       # FID / precision / recall / density / coverage
python scripts/eval_subset.py --help            # DDIM grid of a fixed subset
python scripts/eval_hpa_vae_cypm_recon.py --help
```

All evaluation scripts read their defaults from checkpoint-side args; any
`/home/...` placeholders now require explicit CLI values.

---

## Notes / known limitations

- **This repo is not the upstream DiT.** It ships with the minimum code needed to
  run the experiments described above; many DiT-main utilities (class-conditional
  ImageNet training, the EMA PyTorch DDP sampler, FID inception download, …) have
  been removed or stripped down.
- **`sample_ddp.py` is intentionally removed.** It depended on
  `DiT_models[...](num_classes=...)` which no longer exists and on `torch.distributed`
  bootstrapping specific to the upstream DiT sample.
- The 17 `.sh` runner scripts under `scripts/` are kept as-is for reproducibility;
  their absolute dataset / checkpoint / output variables need to be edited for
  each new machine (variable values are replaced with `/path/to/...` placeholders).
- `config/dataset_cypm.yaml` is optional; if present, train.py reads the CYPM root
  from it and falls back to `SUBDIT_DATA_ROOT` → finally to an empty string that
  forces `--data-root` to be supplied.

---

## Citation

If you use this repository in a publication, please cite the DiT paper and the
CellSeg / SubCell works your REPA teacher and struct encoder originate from.
A paper describing the full SUBdit setup including DFA-CFG and the latent
adversarial objective is in preparation.
