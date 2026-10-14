import os
from math import pi as PI
from math import ceil

import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy import stats

import configparser

plt.rcParams.update({"font.size": 16})

MMU = 105.658 # (MeV) muon mass
MN = 939.56542052
MP = 938.27208943
MNUC = (MN + MP)/2
MEV_TO_GEV = 1e-3
power = 2.0

OUTPUT_DIR = "outputs/"
OUTPUT_FIG = "figures/"
OUTPUT_FILE = "files/"
numugen_string = (
"""_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
                    _____   ______   __   _                                
  _   _            / ____| |  ____| |  \ | |  NuMu-GEN Neutrino Generator  
 / \ ( |     ___  | |  __  | |___   | | \| |  J. L. Bonilla et al.         
  \ \//     |___| | |__\ | |  ___|  | |\ \ |                               
   \_/  |_|        \_____/ |______| |_| \__|  26.10                        
        |                                                                  
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
)



def parse_value_or_list(value, target_type):
    values = [target_type(part) for part in value.split()]
    
    return values



def load_config(parser):
    """Parse config.ini and return conf as a dict."""
    known, _ = parser.parse_known_args()
    cfg = configparser.ConfigParser()
    cfg.read(known.conf)
    
    conf = {}
    
    conf['generator'] = cfg.get('models', 'generator')
    conf['xsec_model'] = cfg.get('models', 'xsec_model')
    
    conf['flux'] = parse_value_or_list(cfg.get('events', 'flux'), float)
    conf['n_events'] = cfg.getint('events', 'n_events')
    
    conf['output'] = cfg.get("outputs", "output")
    conf['no_plots'] = cfg.getboolean("outputs", "no_plots")
    conf['norm'] = cfg.getboolean("outputs", "norm")
    
    conf['batch_size'] = cfg.getint('performance', 'batch_size')
    conf['gpu'] = cfg.getboolean("performance", "gpu")
    
    parser.set_defaults(**conf)
    args = parser.parse_args()
    
    return args



def predict(model, latent_points, Enu, n_events=None, chunk_size=1000):
    from tensorflow.keras.utils import Progbar
    # produce as many events as number of latent vectors
    if n_events is None:
        n_events = Enu.shape[0]
    predictions = np.array([])
    
    steps = ceil(n_events/chunk_size)
    progress = Progbar(steps)
    
    for index in range(0, n_events, chunk_size):
        progress.update(index/chunk_size+1)
        prediction = model(
            [latent_points[index:index+chunk_size],
            Enu[index:index+chunk_size]], training=True
        ).numpy()
        if index == 0:
            predictions = prediction
        else:
            predictions = np.append(predictions, prediction, axis=0)
    
    return predictions



def normalize_Enu(Enu, min_Enu=None, max_Enu=None):
    vEnu = Enu/1000
    vEnu = vEnu + np.exp(1)
    vEnu = np.log(vEnu)
    vEnu = vEnu - np.log(np.exp(1) + min_Enu/1000) - 1
    
    return 2*vEnu



def normalize_Emu(Emu, Enu, lepmass=MMU):
    DeltaE = Enu - lepmass
    groundedE = Emu - lepmass
    Emu = groundedE/DeltaE
    
    for i in range(4):
        Emu = (1 - Emu)**(1.0/power)
    
    Emu = 2*Emu - 1
    
    return Emu



def unnormalize_Emu(vEmu, Enu, lepmass=MMU):
    Emu = (vEmu + 1)/2.0
    
    for i in range(4):
        Emu = 1.0 - Emu**power
    
    Emu = (Enu - lepmass)*Emu + lepmass
    
    return Emu



def normalize_thetamu(thetamu):
    theta = thetamu/PI
    
    for i in range(4):
        theta = (1 - theta)**(1.0/power)
    
    theta = 2*theta - 1
    
    return theta



def unnormalize_thetamu(vthetamu):
    theta = (vthetamu + 1)/2
    
    for i in range(4):
        theta = 1 - theta**power
    
    theta = PI*theta
    
    return theta



def get_enu_dist(flux, bin_edges, nevents=10000, xsec=None, min_val=MMU):
    """
    Get enu distribution given a histogram of flux
    """
    up_min = bin_edges[bin_edges>MMU].min()
    dn_min = bin_edges[bin_edges<=MMU].max()
    
    
    if xsec is not None:
        pdf_bins = flux*xsec
    else:
        pdf_bins = flux.copy()
    
    eff_edges = bin_edges[bin_edges>MMU]
    eff_edges[0] = MMU
    eff_pdf =  pdf_bins[-eff_edges.size+1:]
    
    eff_pdf /= eff_pdf.sum()*np.diff(eff_edges)
    
    var = stats.rv_histogram((eff_pdf, eff_edges), density=True)
    enu = var.rvs(size=nevents)
    enu = enu[enu>min_val]
    
    if enu.size < nevents:
        mevents = nevents - enu.size
        new_enu = get_enu_dist(flux, bin_edges, mevents, xsec, min_val)
        enu = np.append(enu, new_enu)
    
    return enu



def calc_Ehad(Enu, Emu):
    return Enu - Emu



def calc_pmu(Emu, lepmass):
    return np.sqrt(Emu**2 - lepmass**2)



def calc_pmuz(Emu, thetamu, lepmass=MMU):
    return calc_pmu(Emu, lepmass)*np.cos(thetamu)



def calc_pmut(Emu, thetamu, lepmass=MMU):
    return np.sqrt(Emu**2 - lepmass**2)*np.sin(thetamu)



def calc_Q2(Enu, Emu, thetamu, lepmass):
    return 2*Enu*(Emu - calc_pmuz(Emu, thetamu, lepmass)) - lepmass**2



def calc_Invmass(Ehad, Q2, Mnuc=MNUC):
    W2 = Mnuc**2 + 2*Mnuc*Ehad - Q2
    sign = W2/np.abs(W2)
    W = sign*np.sqrt(np.abs(W2))
    
    return W



def make_hist(
        var_gen, bins=100, fig_name="Histogram", arange=None,
        density=False, xlabel=None, ylabel=None, ylogscale=False,
        figure_format="png", label_gen="GAN", xsec=None
    ):
    os.makedirs(OUTPUT_DIR+OUTPUT_FIG, exist_ok=True)
    
    fig, ax = plt.subplots(layout="constrained")
    ax.ticklabel_format(
        style='sci', axis='y', scilimits=(0,0), useOffset=False
    )
    
    histogram, bins = np.histogram(var_gen, bins=bins, range=arange)
    if density or xsec is not None:
        histogram = histogram/histogram.sum()/np.diff(bins)
        if xsec is not None:
            histogram *= xsec
            
    
    ax.hist(
        bins[:-1], bins=bins, weights=histogram, range=arange, histtype='step',
        label=label_gen
    )
    
    ax.legend(loc='upper left', frameon=False)
    plt.xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    ymax *= 6/5
    ax.set_ylim([ymin, ymax])
    
    path = OUTPUT_DIR + OUTPUT_FIG
    print("Saving {} histogram in {}".format(fig_name, path))
    if type(figure_format) is list:
        for form in figure_format:
            plt.savefig(path + fig_name + "." + form)
    else:
        plt.savefig(path + fig_name + "." + figure_format)
    
    plt.close()



def RunNuWroGAN(
        m_xsec, m_generator,
        n_events, in_flux, batch_size,
        output, no_plots, norm
    ):
    print(numugen_string)
    import tensorflow as tf
    bins_2d = [50, 50]
    bins = 50
    
    print("Generating {} events with '{}' and '{}'.".format(
        n_events, m_generator, m_xsec))
    m_name = m_generator.split(".")[0]
    
    lepmass_gen = MMU
    
    xsec_model = tf.keras.models.load_model(m_xsec)
    generator = tf.keras.models.load_model(m_generator)
    
    print("Predicting cross section and sampling initial energies...")
    if len(in_flux) == 1:
        if in_flux[0] <= MMU:
            raise ValueError("Neutrino energies must be larger than muon mass.")
        Enu_gen = in_flux[0]*np.ones(n_events)
        xsec_pred = xsec_model.predict([[in_flux[0], in_flux[0], 0]])
        nn_xsec = in_flux[0]*xsec_pred**2/1000
        nn_xsec = nn_xsec.sum()
    else:
        in_flux = np.array(in_flux)
        flux = in_flux[2:]
        flux /= flux.sum()
        min_enu = in_flux[0]
        max_enu = in_flux[1]
        bin_edges = np.linspace(min_enu, max_enu, flux.size + 1)
        
        enu_bin_mid = (bin_edges[:-1] + bin_edges[1:])/2/1000
        edges_for_gan = np.repeat(bin_edges,2)[1:-1].reshape(-1,2)/2e4
        edges_for_gan = np.append(edges_for_gan, np.diff(edges_for_gan), axis=-1)
        
        xsec_gan_arr = xsec_model.predict(edges_for_gan).flatten()
        xsec_gan_arr = enu_bin_mid*xsec_gan_arr**4
        
        # Get pdf from flux and cross sections
        nn_weighted_flux = xsec_gan_arr*flux
        nn_xsec = nn_weighted_flux.sum()
        nn_pdf = nn_weighted_flux / nn_xsec / np.diff(bin_edges)
        
        Enu_gen = get_enu_dist(nn_pdf, bin_edges, n_events)
        
    
    print("Obtaining muon kinematics...")
    Enu_in = np.reshape(Enu_gen, [Enu_gen.shape[0], 1])
    Enu_in = normalize_Enu(Enu_in, 110.0, 10000.0)
    latent_points = np.random.normal(size=(Enu_gen.shape[0], 50))
    
    gen_events = predict(generator, latent_points, Enu_in, n_events, batch_size)
    
    vEmu_gen = gen_events[:,0]
    vthetamu_gen = gen_events[:,1]
    
    Emu_gen = unnormalize_Emu(vEmu_gen, Enu_gen, lepmass_gen)
    thetamu_gen = unnormalize_thetamu(vthetamu_gen)
    pmuz_gen = calc_pmuz(Emu_gen, thetamu_gen, lepmass_gen)
    Ehad_gen = calc_Ehad(Enu_gen, Emu_gen)
    Q2_gen = calc_Q2(Enu_gen, Emu_gen, thetamu_gen, lepmass_gen)
    Invmass_gen = calc_Invmass(Ehad_gen, Q2_gen)
    
    print('---------------------------------------------------------------------------')
    print(" Flux integrated cross section: {} cm^2".format(nn_xsec*1e-38))
    print('---------------------------------------------------------------------------')
    
    evts = {
        "Enu": Enu_gen,
        "Emu": Emu_gen,
        "theta": thetamu_gen,
        "pmuz": pmuz_gen,
        "Ehad": Ehad_gen,
        "Q2": Q2_gen,
        "W": Invmass_gen
    }
    
    path = OUTPUT_DIR + OUTPUT_FILE
    print("Saving {} files in {}".format(output, path))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(path, exist_ok=True)
    output_name = path + output
    with uproot.recreate(output_name + ".root") as f:
        f.mktree("events", evts)
    with open(output_name + ".txt", 'w') as f:
        f.write("Flux integrated cross section: {} cm^2".format(nn_xsec*1e-38))
    
    if not no_plots:
        xsec = nn_xsec if norm else None
        
        # create dictionaries containing histograms information
        Emu_hist = {}
        Ehad_hist = {}
        thetamu_hist = {}
        pmuz_hist = {}
        Q2_hist = {}
        Invmass_hist = {}
        prob_label = "Events per bin"
        
        # Emu
        Emu_hist["gan"] = Emu_gen * MEV_TO_GEV
        Emu_hist["bins"] = bins
        Emu_hist["range"] = None
        Emu_hist["xlabel"] = r"$E_\mu$ (GeV)"
        Emu_hist["ylabel"] = r"$d\sigma/dE_\mu(\times10^{-38})$" if xsec is not None else prob_label
        
        # Ehad
        Ehad_hist["gan"] = Ehad_gen * MEV_TO_GEV
        Ehad_hist["bins"] = bins
        Ehad_hist["range"] = None
        Ehad_hist["xlabel"] = r"$E_{had}$ (GeV)"
        Ehad_hist["ylabel"] = r"$d\sigma/dE_{had}(\times10^{-38})$" if xsec is not None else prob_label
        
        # Invmass
        Invmass_hist["gan"] = Invmass_gen[Invmass_gen>0] * MEV_TO_GEV
        Invmass_hist["bins"] = bins
        Invmass_hist["range"] = None
        Invmass_hist["xlabel"] = r"$W$ (GeV)"
        Invmass_hist["ylabel"] = r"$d\sigma/dW(\times10^{-38})$" if xsec is not None else prob_label
        
        # thetamu
        thetamu_hist["gan"] = thetamu_gen
        thetamu_hist["bins"] = bins
        thetamu_hist["range"] = [0,PI]
        thetamu_hist["xlabel"] = r"${\theta_\mu}$ (rad)"
        thetamu_hist["ylabel"] = r"$d\sigma/d\theta_\mu(\times10^{-38})$" if xsec is not None else prob_label
        
        # pmuz
        pmuz_hist["gan"] = pmuz_gen * MEV_TO_GEV
        pmuz_hist["bins"] = bins
        pmuz_hist["range"] = None
        pmuz_hist["xlabel"] = r"${p_\mu}_z$ (GeV)"
        pmuz_hist["ylabel"] = r"$d\sigma/dp_\mu(\times10^{-38})$" if xsec is not None else prob_label
        
        # Q2
        Q2_hist["gan"] = Q2_gen * MEV_TO_GEV**2
        Q2_hist["bins"] = bins
        Q2_hist["range"] = None
        Q2_hist["xlabel"] = r"${Q^2}$ (GeV$)^2$"
        Q2_hist["ylabel"] = r"$d\sigma/dQ^2(\times10^{-38})$" if xsec is not None else prob_label
        
        vars_dic = {}
        
        vars_dic["Emu"] = Emu_hist
        vars_dic["Ehad"] = Ehad_hist
        vars_dic["thetamu"] = thetamu_hist
        vars_dic["pmuz"] = pmuz_hist
        vars_dic["Q2"] = Q2_hist
        vars_dic["Invmass"] = Invmass_hist
        
        for var_name, value_dic in vars_dic.items():
            make_hist(
                value_dic["gan"],
                fig_name=output + "_" + var_name,
                bins=value_dic["bins"],
                arange=value_dic["range"],
                xlabel=value_dic["xlabel"],
                ylabel=value_dic["ylabel"],
                figure_format=["pdf", 'png'],
                xsec=xsec
            )
            
    return evts
