# NuMuGen

GAN-based generator for neutrino-Argon interaction events.
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
