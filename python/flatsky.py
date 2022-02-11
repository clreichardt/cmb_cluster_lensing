#import numpy as np, sys, os, scipy as sc
#from scipy.stats import binned_statistic as binstats
import numpy as np
################################################################################################################
#flat-sky routines
################################################################################################################

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2D_cl
    inputs:
    el = el values over which cl is defined
    cl = power spectra - cl

    flatskymyapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    output:
    2d_cl
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl, left = 0., right = 0.).reshape(ell.shape)

    return cl2d

################################################################################################################

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    output:
    lx, ly
    """

    ny, nx, dx = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

################################################################################################################

def get_lxly_az_angle(lx,ly):

    """
    azimuthal angle from lx, ly

    inputs:
    lx, ly = 2d lx and ly arrays

    output:
    azimuthal angle
    """
    return 2*np.arctan2(lx, -ly)

################################################################################################################
def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu = 1):

    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx,ly)

    map1_fft, map2_fft = np.fft.fft2(map1),np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft - np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real
    else:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft + np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( -np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real

    return map1_mod, map2_mod
################################################################################################################

def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 'lowpass'):
    """
    These are step function filters -- 
    filter_type = 0/'lowpass' - low pass filter
    filter_type = 1/'highpass' - high pass filter
    filter_type = 2/'bandpass' - band pass
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0 or filter_type == 'lowpass':
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1 or filter_type == 'highpass':
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2 or filter_type == 'bandpass':
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    fft_filter[np.isnan(fft_filter)] = 0.
    fft_filter[np.isinf(fft_filter)] = 0.

    return fft_filter
################################################################################################################

def wiener_filter(flatskymapparams, cl_signal, cl_noise, el = None):
    #note any ells outside provided el's will be set to zero for
    
    if el is None:
        el = np.arange(len(cl_signal))

    ny, nx, dx = flatskymapparams

    #prep 1d version
    wiener = cl_signal / (cl_signal + cl_noise)
    wiener[cl_signal + cl_noise <= 0] =  0.0
    wiener[np.isnan[wiener]]=0.0
    wiener[np.isinf[wiener]]=0.0
    
    #now throw it to 2d
    wiener_filter = cl_to_cl2d(el, wiener, flatskymapparams)

    return wiener_filter

################################################################################################################

def map2cl(mask, pixel_arcmin, flatskymap1, flatskymap2 = None, binsize = None, filter_2d = None):

    """
    map2cl module - get the power spectra of map/maps

    input:
    pixel_arcmin: resolution in arcminutes
    
    mask: larger mask file. the input maps will be expanded to these dimensions

    flatskymap1: map1. Must be smaller or equal in both dimensions to mask
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    dx_rad = np.radians(pixel_arcmin/60.)
    nx,ny = mask.shape
    fsky = np.mean(mask)
    prefactor = (dx_rad**2)/(nx*ny)/fsky

    loc_filter2d = 1        
    if filter_2d is not None:
        mx,my = filter_2d.shape
        assert(mask.shape == filter_2d.shape)
        loc_filter2d = filter_2d

    mx,my = flatskymap1.shape
    assert (mx <= nx and my <= ny)
    loc_map = mask.copy()
    loc_map[0:mx,0:my] *= flatskymap1   
    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * loc_filter2d)**2
    else: #cross spectra now
        assert( flatskymap1.shape == flatskymap2.shape )
        loc_map2 = mask.copy()
        loc_map2[0:mx,0:my] *= flatskymap2
        flatskymap_psd = np.fft.fft2(loc_map2) * np.conj(np.fft.fft2(loc_map1))
        
    lx, ly = get_lxly([nx,ny,pixel_arcmin])

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]
    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)

    return rad_prf[:,0], rad_prf[:,1]*prefactor



def old_map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, mask = None, filter_2d = None):

    """
    map2cl module - get the power spectra of map/maps

    input:
    flatskymyapparams = [ny, nx, dx] where ny, nx = flatskymap.shape; and dx is the pixel resolution in arcminutes.
    for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = 0.5 arcminutes.

    flatskymap1: map1 with dimensions (ny, nx)
    flatskymap2: provide map2 with dimensions (ny, nx) cross-spectra

    binsize: el bins. computed automatically if None

    cross_power: if set, then compute the cross power between flatskymap1 and flatskymap2

    output:
    auto/cross power spectra: [el, cl, cl_err]
    """

    ny, nx, dx = flatskymapparams
    dx_rad = np.radians(dx/60.)

    lx, ly = get_lxly(flatskymapparams)

    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * dx_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * dx_rad * np.conj( np.fft.fft2(flatskymap2) ) * dx_rad / (nx * ny)

    rad_prf = radial_profile(flatskymap_psd, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
    el, cl = rad_prf[:,0], rad_prf[:,1]

    if mask is not None:
        fsky = np.mean(mask)
        cl /= fsky

    if filter_2d is not None:
        rad_prf_filter_2d = radial_profile(filter_2d, (lx,ly), bin_size = binsize, minbin = 100, maxbin = 10000, to_arcmins = 0)
        el, fl = rad_prf_filter_2d[:,0], rad_prf_filter_2d[:,1]
        cl /= fl

    return el, cl

################################################################################################################

def radial_profile(z, xy = None, binarr = None, bin_size = 1., minbin = 0., maxbin = 10., to_arcmins = True, request_std=False):

    """
    get the radial profile of an image (could be real and fourier space) - only keeps real parts
    returns N x 3 array
    :,0 == bin centres
    :,1 == mean value
    :,2 == std error on mean

    if binarr exists, uses that (no error checking). 
    otherwise creates uniform binning:
    (minbin <= x < minbin+binsize), ....
    Note -- the maxium bin edge may not be maxbin if (maxbin-minbin)/bin_size is not integer. It could be higher or lower. 
    bin_size and minbin are fixed
    """

    z = np.asarray(z)
    if xy is None:
        x, y = np.indices(z.shape)
    else:
        x, y = xy
                                                         
    #radius = np.hypot(X,Y) * 60.
    radius = (x**2. + y**2.) ** 0.5
                                                         
    if to_arcmins: radius *= 60.

    if binarr is None:
        binarr=np.arange(minbin,maxbin+bin_size/2,bin_size) # this gaurantees lower edge. Upper edge may shift if not integer multiple


    bin_numerator, _, _ = binstats(radius,  z, statistic='sum',
                                   bins=binarr,range=[binarr[0],binarr[-1]])
    bin_hits, _, _      = binstats(radius,  z, statistic='count',
                                   bins=binarr,range=[binarr[0],binarr[-1]])
    if request_std:
        bin_std, _, _   = binstats(radius,  z, statistic='std',
                                   bins=binarr,range=[binarr[0],binarr[-1]])

    radprf=np.zeros([len(bin_hits),3],dtype=np.float32)
    use = bin_hits > 0
    radprf[use,1]=bin_numerator[use]/bin_hits[use]
    if request_std:
        radprf[use,2]=bin_std[use]/np.sqrt(bin_hits[use])
    radprf[:,0] = 0.5*(binarr[:-1]+binarr[1:])
    
    return radprf


############################################
def figure_fft_size(dims):
    try:
        n = max(dims) # doing square
    except TypeError:
        n=dims #presumably int
    logn  = np.log(n)
    log2  = np.log(2)
    ratio = logn/log2
    pow2  = np.int(ratio) # +1 to zeropad x 2
    resid = ratio-pow2
    pow2 += 1
    # assert pow2 > 4
    #go back to raw log
    resid *= log2
    #print(n,ratio,resid)
    if resid < 0.0001: # close enough to x2
        return 1 << pow2
    if resid <=  np.log(1.125):
        return (9 <<(pow2-3)) 
    if resid <=  np.log(1.25):
        return (5 <<(pow2-2))
    if resid <=  np.log(1.3125):
        return (21 <<(pow2-4)) 
    if resid <= np.log(1.5):
        return 3 << (pow2-1)
    if resid <=  np.log(1.6875):
        return (27 <<(pow2-4)) 
    if resid <= np.log(1.75):
        return 7 << (pow2-2)
    if resid <= np.log(1.875):
        return 15 << (pow2-3)
    return 1 << pow2+1
    

################################################################################################################

def make_gaussian_realisation(mapparams, el, cl, cl2 = None, cl12 = None, cltwod=None, tf=None, bl = None, qu_or_eb = 'qu'):

    

    ny, nx, dxin = mapparams
    use_n = figure_size([nx,ny])
    #this is the zero-padded size
    fft_mapparams = [use_n, use_n, dxin]
    
    arcmins2radians = np.radians(1/60.)

    dx = dxin * arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx**2.))
    ################################################

    # Error checking
    if cltwod is not None and cltwod.shape != (use_n,use_n):
        raise ValueError('cltwod provided, but is wrong shape - need square array dim = {}'.format(use_n))
    if tf is not None:
        if isinstance(tf, np.ndarray) and tf.shape != (use_n,use_n):
            raise ValueError('tf provided, but is wrong shape - need square array dim = {}'.format(use_n))
        else:
            if tf['T'].shape != (use_n, use_n) or tf['E'].shape != (use_n, use_n):
                raise ValueError('tf[T/E] provided, but one is wrong shape - need square array dim = {}'.format(use_n))
    #if cltwod is given, directly use it, otherwise do 1d to 2d
    
    if cltwod is None 
        cltwod = cl_to_cl2d(el, cl, fft_mapparams)

    # if the tranfer function is given, correct the 2D cl by tf
    if tf is not None:
        if isinstance(tf, np.ndarray):
            cltwod = cltwod * tf**2
        else:
            cltwod = cltwod * tf['T']**2

    ################################################
    if cl2 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, mapparams)
        cltwod2 = cl_to_cl2d(el, cl2, mapparams)
        if tf is not None:
            cltwod2 = cltwod2 * tf['E']**2
            cltwod12 = cltwod12 * tf['T'] * tf['E']

    ################################################
    if cl2 is None:

        cltwod = cltwod**0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([nx,ny])
        SIM = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        assert qu_or_eb in ['qu', 'eb']

        cltwod[np.isnan(cltwod)] = 0.
        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod2[np.isnan(cltwod2)] = 0.

        #in this case, generate two Gaussian random fields
        #SIM_FIELD_1 will simply be generated from gauss_reals_1 like above
        #SIM_FIELD_2 will generated from both gauss_reals_1, gauss_reals_2 using the cross spectra
        gauss_reals_1 = np.random.standard_normal([nx,ny])
        gauss_reals_2 = np.random.standard_normal([nx,ny])

        gauss_reals_1_fft = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2_fft = np.fft.fft2( gauss_reals_2 )

        #field_1
        cltwod_tmp = np.copy( cltwod )**0.5 * norm
        SIM_FIELD_1 = np.fft.ifft2( cltwod_tmp *  gauss_reals_1_fft ).real
        #SIM_FIELD_1 = np.zeros( (ny, nx) )

        #field 2 - has correlation with field_1
        t1 = np.copy( gauss_reals_1_fft ) * cltwod12 / np.copy(cltwod)**0.5
        t2 = np.copy( gauss_reals_2_fft ) * ( cltwod2 - (cltwod12**2. /np.copy(cltwod)) )**0.5
        SIM_FIELD_2_FFT = (t1 + t2) * norm
        SIM_FIELD_2_FFT[np.isnan(SIM_FIELD_2_FFT)] = 0.
        SIM_FIELD_2 = np.fft.ifft2( SIM_FIELD_2_FFT ).real

        #T and E generated. B will simply be zeroes.
        SIM_FIELD_3 = np.zeros( SIM_FIELD_2.shape )
        if qu_or_eb == 'qu': #T, Q, U: convert E/B to Q/U.
            SIM_FIELD_2, SIM_FIELD_3 = convert_eb_qu(SIM_FIELD_2, SIM_FIELD_3, mapparams, eb_to_qu = 1)
        else: #T, E, B: B will simply be zeroes
            pass

        SIM = np.asarray( [SIM_FIELD_1, SIM_FIELD_2, SIM_FIELD_3] )

    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, mapparams)
        if cl2 is None:
            SIM = np.fft.ifft2( np.fft.fft2(SIM) * bl).real
        else:
            for tqu in range(len(SIM)):
                SIM[tqu] = np.fft.ifft2( np.fft.fft2(SIM[tqu]) * bl).real

    if cl2 is None:
        SIM = SIM - np.mean(SIM)
    else:
        for tqu in range(len(SIM)):
            SIM[tqu] = SIM[tqu] - np.mean(SIM[tqu])

    return SIM

################################################################################################################
