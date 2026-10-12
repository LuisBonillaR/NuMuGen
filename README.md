# NuMuGen


GAN-based generator for neutrino-Argon interaction events based on the paper: 
- __Machine-learning-based neutrino-event generation constrained by experimental measurements__, by Jose L. Bonilla, Krzysztof M. Graczyk, Rwik Dharmapal Banerjee, Beata E. Kowal, Hemant Prasad, Jan T. Sobczyk, [arXiv:2610.xxxxx](xx)
- Abstract: We present a generative surrogate for charged-current \(\nu_\mu\)-nucleus scattering that can be calibrated to measured differential cross sections without event-level experimental samples. Starting from a theoretical interaction model, we fine-tune the surrogate to a MicroBooNE measurement and show that this also improves the description of observables not used in the fit. The calibrated model can also be applied to measurements from different neutrino fluxes without retuning, while retaining event-level predictions of outgoing-muon kinematics. Our results show that experimental cross sections can directly constrain generative event models while preserving their applicability across different observables and beam configurations. 

<img width="640" height="480" alt="NuMu_Gen_Invmass" src="https://github.com/user-attachments/assets/3a8d8dae-74b7-4859-a81f-cb2506f83df5" />

## Installation

```bash
conda env create -f environment.yml
conda activate numugen
```

## Usage

**From the command line:**

```bash
# Default options
python run_numugen.py

# 200000 events from monoenergetic beam (MeV)
python run_numugen.py --n-events 200000 --flux 1000

# From default config file
python run_numugen.py --config configs/custom.ini
```

## Configuration

Create or edit `configs/default.ini` to set model paths, event counts, GPU usage, etc:

```ini
[models]
generator  = ; Path to generator .keras/.h5 model
xsec_model = ; Path to cross section .keras/.h5 model

[events]
flux     = ; Flux. Either monoenergetic beam or flux min_enu max_enu bins_height
n_events = ; Number of events to be produced

[plots]
no_plots = ; bool. Do not save plots
norm     = ; Scale plots by predicted flux integrated cross section

[performance]
batch_size = ; Step size to produce n_events
gpu        = ; bool. Use GPU (when available)
```

CLI flags always override config file values.

# Acknowledgments

This research is partly (K.M.G., B.E.K, J.T.S.) or fully (J.L.B, H.P., R.D.B.) supported by the National Science Centre under grant UMO-2021/41/B/ST2/02778. J.L.B. and J.T.S. are also supported by the Polish Ministry of Science Grant No. 2022/WK/15.
