import os
import argparse

from utils.numugan_utils import load_config, RunNuWroGAN

parser = argparse.ArgumentParser()
parser.add_argument('--conf', default='config/default.ini')
parser.add_argument('--generator', type=str,
                    help='GAN generator name.')
parser.add_argument('--xsec-model', type=str,
                    help='sigma-DNN name.')
parser.add_argument(
    '--n-events', type=int,
    help="Number of events per batch to be produced. Defaults to 1000"
)
parser.add_argument(
    '--batch-size', type=int,
    help="Number of events to be produced. Defaults to 200000"
)
parser.add_argument(
    '--flux', type=float, nargs="*",
    help='Energy of neutrino beam in MeV. Defaults to 1000 MeV.'
)
parser.add_argument(
    "--no-plots", action="store_true", default=None,
    help="Bool. Use it to prevent saving histograms of generated variables."
)
parser.add_argument(
    "--norm", action="store_true", default=None,
    help="Reescale plots by predicted cross section."
)
parser.add_argument(
    "--gpu", action="store_true", default=None,
    help="Bool. Use GPU instead of CPU for the generator, if available."
)



if __name__ == "__main__":
    args = load_config(parser)
    if not args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    xsec_model = args.xsec_model
    generator = args.generator
    n_events = args.n_events
    flux = args.flux
    no_plots = args.no_plots
    no_plots = args.no_plots
    norm = args.norm
    batch_size = args.batch_size
    
    RunNuWroGAN(
        xsec_model,
        generator,
        n_events,
        flux,
        batch_size,
        no_plots,
        norm
    )
    


