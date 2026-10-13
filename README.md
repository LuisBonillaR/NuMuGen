# NuMuGen

NuMuGen is a lightweight inference package for the tuned neural surrogate
introduced in the paper below. It generates charged-current
$\nu_\mu$-argon interaction events by combining:

- a conditional GAN that samples outgoing-muon kinematics at fixed neutrino
  energy, and
- a cross-section DNN that supplies the energy-dependent normalization and
  flux weighting.

The program takes either a monoenergetic neutrino beam or a binned neutrino
flux as input, samples flux-weighted incoming neutrino energies, generates
event-level muon kinematics, and writes histograms of derived observables such
as $E_\mu$, $E_{had}$, $\theta_\mu$, $p_{\mu z}$, $Q^2$, and the
hadronic invariant mass $W$. With normalization enabled, the histograms are
scaled by the predicted flux-integrated cross section.

This repository contains the post-tuned models used for public
prediction/reproduction studies. It is not the full training or fine-tuning
pipeline: the NuWro training samples, MicroBooNE covariance matrices, smearing
matrices, and reservoir-construction code are not included.

The main result of the associated work is that a GAN event generator can be
constrained directly by measured differential cross sections while retaining
event-level predictions and remaining usable for other flux configurations.
The tuned model distributed here is the resulting surrogate.

Based on:

- __Machine-learning-based neutrino-event generation constrained by experimental measurements__, by Jose L. Bonilla, Krzysztof M. Graczyk, Rwik Dharmapal Banerjee, Beata E. Kowal, Hemant Prasad, and Jan T. Sobczyk, [arXiv:2610.xxxxx](xx).

<img width="640" height="480" alt="NuMu_Gen_Invmass" src="https://github.com/user-attachments/assets/3a8d8dae-74b7-4859-a81f-cb2506f83df5" />

## Installation

```bash
conda env create -f environment.yml
conda activate numugen
```

The distributed `.keras` models were saved with Keras/TensorFlow 2.15.
Use the environment above, or an equivalent environment with TensorFlow 2.15,
to avoid model deserialization errors with newer Keras versions.

## Usage

**From the command line:**

```bash
# Default options
python run_numugen.py

# 200000 events from monoenergetic beam (MeV)
python run_numugen.py --n-events 200000 --flux 1000

# From a custom config file
python run_numugen.py --conf config/custom.ini

# Do not write histograms
python run_numugen.py --no-plots
```

## Configuration

Create or edit `config/default.ini` to set model paths, event counts, GPU usage, etc:

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
