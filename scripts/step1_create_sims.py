#!/usr/bin/env python
########################

########################
#load desired modules
import numpy as np, sys, os, scipy as sc, argparse
sys_path_folder='/Users/sraghunathan/Research/SPTPol/analysis/git/cmb_cluster_lensing/python/'
sys.path.append(sys_path_folder)

import flatsky, tools, lensing, foregrounds, misc

from tqdm import tqdm

from pylab import *
cmap = cm.RdYlBu_r

import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
print('\n')
########################

########################
parser = argparse.ArgumentParser(description='')
parser.add_argument('-start', dest='start', action='store', help='start', type=int, default=0)
parser.add_argument('-end', dest='end', action='store', help='end', type=int, default=10)
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type=str, required=True)#='params.ini')
parser.add_argument('-clusters_or_randoms', dest='clusters_or_randoms', action='store', help='clusters_or_randoms', type=str, default='clusters')
parser.add_argument('-random_seed_for_sims', dest='random_seed_for_sims', action='store', help='random_seed_for_sims', type=int, default=-1)#111)

args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]

    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)

if clusters_or_randoms == 'randoms':
    start, end = 0, 1

########################

########################
print('\tread/get necessary params')
param_dict = misc.get_param_dict(paramfile)

data_folder = param_dict['data_folder']
results_folder = param_dict['results_folder']

#params or supply a params file
dx = param_dict['dx'] #pixel resolution in arcmins
@boxsize_am = param_dict['boxsize_am'] #boxsize in arcmins
nx = param_dict['boxsize_nx'] #boxsize in pixels
nfft = 2*nx
boxsize_am = nx*dx
#nx = int(boxsize_am/dx)
mapparams    = [nx,   nx,   dx]
fftmapparams = [nfft, nfft, dx]
assert( int(nx/2)*2 == nx) #odd numbers not dealt with.
x1,x2 = -nx/2. * dx, nx/2. * dx
verbose = 0
pol = param_dict['pol']
debug = param_dict['debug']

#beam and noise levels
noiseval = param_dict['noiseval'] #uK-arcmin
if pol:
    noiseval = [noiseval, noiseval * np.sqrt(2.), noiseval * np.sqrt(2.)]
beamval = param_dict['beamval'] #arcmins

#foregrounds
try:
    fg_gaussian = param_dict['fg_gaussian'] #Gaussian realisation of all foregrounds
except:
    fg_gaussian = False

try:
    add_cluster_tsz=param_dict['add_cluster_tsz']
except:
    add_cluster_tsz=False

try:
    add_cluster_ksz=param_dict['add_cluster_ksz']
except:
    add_cluster_ksz=False

try:
    pol_frac_radio = param_dict['pol_frac_radio']
except:
    pol_frac_radio = False

try:
    pol_frac_cib = param_dict['pol_frac_cib']
except:
    pol_frac_cib = False

#ILC
try:
    ilc_file = param_dict['ilc_file'] #ILC residuals
    which_ilc = param_dict['which_ilc']
except:
    ilc_file = None
    which_ilc = None

if ilc_file is not None:
    fg_gaussian = None
    if which_ilc == 'cmbtszfree':
        add_cluster_tsz = None
    else:
        print('\n\n\tyou have requested a ILC that is not tSZ-free. Weighted tsz is not implemented yet. aborting script here.')
        sys.exit()

#CMB power spectrum
cls_file = '%s/%s' %(param_dict['data_folder'], param_dict['cls_file'])

if not pol:
    tqulen = 1
else:
    tqulen = 3
tqu_tit_arr = ['T', 'Q', 'U']

#Map properties:
try:
    iso_ell_hpf = param_dict['iso_ell_hpf']
except:
    iso_ell_hpf = 400.
try:
    x_ell_hpf = param_dict['x_ell_hpf']
except:
    x_ell_hpf = 400.

#apod properties:
try:
    edge_taper_arcmin = param_dict['edge_taper_arcmin']
except: 
    edge_taper_arcmin = 10.0 #arcmin
small_apod_mask = tools.get_apod_mask(mapparams,edge_taper_arcmin)

#sim stuffs
#total_sim_types = param_dict['total_sim_types'] #unlensed background and lensed clusters
total_clusters = param_dict['total_clusters']
total_randoms = param_dict['total_randoms'] #total_clusters * 10 #much more randoms to ensure we are not dominated by variance in background stack.

#cluster info
cluster_mass = param_dict['cluster_mass']
cluster_z = param_dict['cluster_z']

#cluster mass definitions
delta=param_dict['delta']
rho_def=param_dict['rho_def']
profile_name=param_dict['profile_name']

#cosmology
#h=param_dict['h']
#omega_m=param_dict['omega_m']
#omega_lambda=param_dict['omega_lambda']
#z_lss=param_dict['z_lss']
#T_cmb=param_dict['T_cmb']

#cutouts specs 
cutout_size_am = param_dict['cutout_size_am'] #arcmins

#for estimating cmb gradient
apply_wiener_filter = param_dict['apply_wiener_filter']
lpf_gradient_filter = param_dict['lpf_gradient_filter']
cutout_size_am_for_grad = param_dict['cutout_size_am_for_grad'] #arcminutes
########################

########################
#get ra, dec or map-pixel grid
xvec=np.linspace(-0.5*dx/60.*(nx-1),
                  0.5*dx/60.*(nx-1),
                  , nx) #degrees

x_grid_deg, y_grid_deg=np.meshgrid(xvec,xvec)
########################

########################
#CMB power spectrum - read Cls now
if not pol:
    tqulen=1
else:
    tqulen=3
tqu_tit_arr=['T', 'Q', 'U']

el, cl = tools.get_cmb_cls(cls_file, pol = pol)
########################

########################
#get beam and noise
bl = tools.get_bl(beamval, el, make_2d = 1, mapparams = fftmapparams)

#noise
if ilc_file is None:
    nl_dic = tools.get_nl_dic(noiseval, el, pol = pol)
else:
    ilc_dic = np.load(ilc_file, allow_pickle = True).item()
    weights_arr, cl_residual_arr = ilc_dic['TT'][which_ilc]
    cl_residual_arr = np.interp(el, np.arange(len(cl_residual_arr)), cl_residual_arr)
    nl_dic = {}
    nl_dic['T'] = cl_residual_arr
print('\tkeys in nl_dict = %s' %(str(nl_dic.keys())))
########################

########################
#get foreground spectra if requested
if fg_gaussian:
    cl_fg_dic = tools.get_cl_fg(el = el, freq = 150, pol = pol, pol_frac_cib = pol_frac_cib, pol_frac_radio = pol_frac_radio)
########################

########################
#plot
if debug:
    ax =subplot(111, yscale='log', xscale='log')
    plot(el, cl[0], color='black', label=r'TT')
    plot(el, nl_dic['T'], color='black', ls ='--', label=r'Noise: T')
    if fg_gaussian:
        plot(el, cl_fg_dic['T'], color='black', ls ='-.', label=r'Foregrounds: T')
    if pol:
        plot(el, cl[1], color='orangered', label=r'EE')
        plot(el, nl_dic['P'], color='orangered', ls ='--', label=r'Noise: P')
        if fg_gaussian:
            plot(el, cl_fg_dic['P'], color='orangered', ls ='-.', label=r'Foregrounds: P')
    legend(loc=1)
    ylim(1e-10, 1e4)
    ylabel(r'$C_{\ell}$ [$\mu$K$^{2}$]')
    xlabel(r'Multipole $\ell$')
    show()
########################


########################
# Loop over cluster positions, randoms or simulated clusters
for index in range(start,end):
    #this could be a cut from the data; a cutout at random loc from the data, a simulated cluster loc, or a simulated random loc
    #local_mask will be all ones, unless there is a pt source that is masked. It will be multiplied by the field apod mask before the fft

    print('consider doing lensing on central region only')
    large_cutout, local_mask = tools.get_big_cutout(mapparams,index,type=data_type)




    #get median gradient direction and magnitude for all cluster cutouts + rotate them along median gradient direction.
    grad_mag, grad_orien, cutout_rotated = tools.get_rotated_tqu_cutout(large_cutout, local_mask, small_apod_mask, mapparams, fftmapparams,
                                                                        cutout_size_am, cutout_size_am_for_grad, 
                                                                        perform_random_rotation = perform_random_rotation, 
                                                                        apply_fft_filter = gradient_filter )
    #    grad_mag_arr, grad_orien_arr, cutouts_rotated_arr = tools.get_rotated_tqu_cutouts(sim_arr, sim_arr, nclustersorrandoms, tqulen, mapparams, cutout_size_am, 
    #                                                                                           apply_wiener_filter=apply_wiener_filter, cl_signal = cl_signal_arr, cl_noise = cl_noise_arr, 
                                                                                            lpf_gradient_filter = lpf_gradient_filter, cutout_size_am_for_grad = cutout_size_am_for_grad)

    if save_intermediate:
        output_dic[data_type]['map'][index]=np.asarray( large_cutout )
        output_dic[data_type]['local_mask'][index]=np.asarray( local_mask )   
        output_dic[data_type]['cutouts_rotated'][index]=cutouts_rotated_arr
        output_dic[data_type]['grad_mag'][index]=grad_mag_arr    

    stack += cutout_rotated * grad_mag
    stack_wt += grad_mag

stack /= stack_wt

output_dic[data_type]['stack']=stack
