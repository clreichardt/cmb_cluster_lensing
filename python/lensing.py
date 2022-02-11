import numpy as np, sys, os, flatsky
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_defs
cosmology.setCosmology('planck15')

from scipy import interpolate as intrp

from pylab import *

#################################################################################
#################################################################################
#################################################################################

def get_deflection_angle_from_convergence(kappa, mapparams,zero_pad = True):
    

    ny, nx, dx_arcmin = mapparams
    if zero_pad:
        use_n = figure_size([nx,ny])
        #this is the zero-padded size
        fft_mapparams = [use_n, use_n, dx_arcmin]
        loc_kappa = np.zeros([use_n,use_n])
        loc_kappa[:ny,:nx]=kappa
    else: 
        fft_mapparams = mapparams
        loc_kappa = kappa
        
    lx, ly = flatsky.get_lxly(fft_mapparams)

    dx_rad = np.radians(dx_arcmin/60.)
    phi_fft = -2. * dx_rad * dx_rad * np.fft.fft2(loc_kappa)/(lx**2 + ly**2)
    phi_fft[np.isnan(phi_fft)] = 0.
    phi_fft[np.isinf(phi_fft)] = 0.

    def_x    = np.fft.ifft2(-1j * phi_fft * lx) / ( dx_rad * dx_rad )
    def_y    = np.fft.ifft2(-1j * phi_fft * ly) / ( dx_rad * dx_rad )


    return def_x[:ny,:nx], def_y[:ny,:nx]

def perform_lensing(theta_x_grid_deg, theta_y_grid_deg, image, kappa, mapparams, poly_deg = 5):
    # no error checking, but assume sizes all match
    theta_x_grid, theta_y_grid = np.radians(theta_x_grid_deg), np.radians(theta_y_grid_deg)
    def_x, def_y = get_deflection_angle_from_convergence(kappa, mapparams)
    mod_theta_x_grid = (theta_x_grid + def_x).flatten().real
    mod_theta_y_grid = (theta_y_grid + def_y).flatten().real

    image_lensed = intrp.RectBivariateSpline( theta_y_grid[:,0], theta_x_grid[0,:], image, kx = poly_deg, ky = poly_deg).ev(mod_theta_y_grid, mod_theta_x_grid).reshape(mapparams[:2])

    return image_lensed

def get_rv(cosmo, Mdelta, z, h, delta, rho_def):

    try:
        if not Mdelta.unit == u.Msun:
            Mdelta = Mdelta*u.Msun
    except:
        Mdelta = Mdelta*u.Msun

    if (rho_def == 'crit'):
        rho_c_z = cosmo.critical_density(z)
    elif (rho_def == 'mean'):
        rho_c_z = cosmo.Om(z)*cosmo.critical_density(z)
    else:
        print("rho definition not specified correctly in cluster profile")
        assert(0)
        rho_c_z = rho_c_z.to('M_sun/Mpc3') #convert critical density into M_sun/MPc3

    r_v = (((Mdelta/(delta*4.*np.pi/3.))/rho_c_z)**(1./3.)).to('Mpc')

    return r_v

#################################################################################

def get_nfw_kappa_deflection_angle(cosmo, theta, Mdelta, z, h, delta, rho_def, profile_name, z_source,deflection=True):

    """
    returns convergence and deflection angle vectors. Currently only supports NFW.
    """

    Mdelta = Mdelta*u.Msun
    if (rho_def == 'crit'):
        rho_c_z = cosmo.critical_density(z)
    elif (rho_def == 'mean'):
        rho_c_z = cosmo.Om(z)*cosmo.critical_density(z)
    else:
        print("rho definition not specified correctly in cluster profile")
        assert(0)
        rho_c_z = rho_c_z.to('M_sun/Mpc3') #convert critical density into M_sun/MPc3

    #r_v = (((Mdelta/(delta*4.*np.pi/3.))/rho_c_z)**(1./3.)).to('Mpc')
    r_v = get_rv(cosmo, Mdelta, z, h, delta, rho_def)

    cdelta = concentration.concentration(Mdelta.value, '%s%s' %(delta, rho_def[0]), z)
    
    if profile_name == 'NFW':

        delta_c = (delta/3.)*(cdelta**3.)/(np.log(1.+cdelta)-cdelta/(1.+cdelta))
        r_s = r_v.to('Mpc')/cdelta

        #Angular diameter distances
        distance_lens = cosmo.comoving_distance(z)/(1.+z)
        distance_source = cosmo.comoving_distance(z_source)/(1.+z_source)
        distance_lens_source = (cosmo.comoving_distance(z_source)-cosmo.comoving_distance(z))/(1.+z_source)
        sigma_c = (((const.c.cgs**2.)/(4.*np.pi*const.G.cgs))*(distance_source/(distance_lens*distance_lens_source))).to('M_sun/Mpc2')

        R = distance_lens*theta
        x = R/r_s

        if (1): #convergence
            g_theta = np.zeros(x.shape)
            gt_one = np.where(x > 1.0)
            lt_one = np.where(x < 1.0)
            eq_one = np.where(np.abs(x - 1.0) < 1.0e-5)
            g_theta[gt_one] = (1./(x[gt_one]**2. - 1))*(1. - (2./np.sqrt(x[gt_one]**2. - 1.))*np.arctan(np.sqrt((x[gt_one]-1.)/(x[gt_one]+1.))).value)
            g_theta[lt_one] = (1./(x[lt_one]**2. - 1))*(1. - (2./np.sqrt(1. - x[lt_one]**2.))*np.arctanh(np.sqrt((1. - x[lt_one])/(x[lt_one]+1.))).value)
            g_theta[eq_one] = 1./3.

            #Projected mass
            sigma = ((2.*r_s*delta_c*rho_c_z)*g_theta).to('M_sun/Mpc2')
            #Final kappa
            kappa = sigma/sigma_c

        if deflection: #deflection angle is requested
            x = x.value

            h_theta = np.zeros(x.shape)
            gt_one = np.where(x > 1.0)
            lt_one = np.where(x < 1.0)
            eq_one = np.where(np.abs(x - 1.0) < 1.0e-5)
            h_theta[gt_one] = 1./x[gt_one] * ( np.log(x[gt_one]/2.) + (2./np.sqrt(x[gt_one]**2. - 1.) * np.arctan( np.sqrt((x[gt_one]-1.)/(x[gt_one]+1.)) ) ) )
            h_theta[lt_one] = 1./x[lt_one] * ( np.log(x[lt_one]/2.) + (2./np.sqrt(1. - x[lt_one]**2.) * np.arctanh( np.sqrt((1. - x[lt_one])/(x[lt_one]+1.)) ) ) )
            h_theta[eq_one] = 1./x[eq_one] * ( np.log(x[eq_one]/2.) )

            A = (Mdelta * cdelta**2.)/(np.log(1.+cdelta)-cdelta/(1.+cdelta))/4/np.pi
            def_angle_vector = (16 * np.pi * const.G.cgs * A.to('g') / cdelta/ const.c.cgs.to('cm/s')**2. / r_v.to('cm')) * (distance_lens_source/distance_source) * h_theta        
            return kappa.value, def_angle_vector.value  
        else:
            return kappa.value, None

#################################################################################

def get_convergence(ra_grid, dec_grid, ra_list, dec_list, mass_list, z_list, param_dict, bl = None):

    cosmo = FlatLambdaCDM(H0 = param_dict['h']*100., Om0 = param_dict['omega_m'])
    map_coords = coord.SkyCoord(ra = ra_grid*u.degree, dec = dec_grid*u.degree)

    if len( np.unique(mass_list) ) == 1 and len( np.unique(z_list) ) == 1:
        tot_iter = 1
    else:
        tot_iter = len(ra_list)

    kappa_arr = []
    for i in range(tot_iter):
        #print(i, end = ' ')
        ra, dec, Mdelta, z = ra_list[i], dec_list[i], mass_list[i], z_list[i]
        if Mdelta == 0.: Mdelta = 1e5
        if i == 0:
            cluster_coords = coord.SkyCoord(ra = ra*u.degree, dec = dec*u.degree)
            theta_grid = map_coords.separation(cluster_coords).value*(np.pi/180.)

        #get nFW convergence
        kappa, _ = get_nfw_kappa_deflection_angle(cosmo, theta_grid, Mdelta, z, param_dict['h'], delta = param_dict['delta'], rho_def = param_dict['rho_def'], profile_name = param_dict['profile_name'], z_source = param_dict['z_lss'],deflection=False)
        kappa_arr.append( kappa )

    if len( np.unique(mass_list) ) == 1 and len( np.unique(z_list) ) == 1:
        kappa_arr = np.repeat( kappa_arr, len(mass_list), axis = 0 )

    return np.asarray( kappa_arr )

