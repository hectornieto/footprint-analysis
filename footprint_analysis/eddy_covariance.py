from osgeo import gdal
from os.path import exists
import numpy as np
from numpy import ma
from scipy.ndimage.interpolation import rotate
from pyTSEB import MO_similarity as mo
from matplotlib import pyplot as plt

# Global constants
# von Karman's constant
KARMAN = 0.4
N_BINS = 1e3
CP = 1005  # Specific heat capacity of dry air at


def footprint_Detto(ustar, h, le, t_a, zm, zo, sigma_v=None, bl_height=3000,
                    cutout=0.9, n_bins=N_BINS):
    '''Estimates the 2D footprint (or source weight function)

    Parameters
    ----------
    ustar : friction velocity (m/s)
    H     : Sensible heat (W/m2)
    LE     : Latent heat (W/m2)
    Ta    : Mean air temperature (oC)
    zm    : Instrument height (m)
    zo    : Momentum roughness height (m)
    sigma_v : standard deviation of the cross wind velocity

    Returns
    -------
    Fp_xy : 2D Source contribution with distance (pixels)
    xy  : Distance vector used in the computation of Fc, fp (m)'''

    # calculate one-dimensional footprint in the upwind direction following Hsieh
    [fc, fp, x, ol, *_] = footprint_Hsieh(ustar, h, le, t_a, zm, zo, n_bins=n_bins)
    if not sigma_v:
        sigma_v = calc_sigma_v(ustar, ol, bl_height=bl_height)

    # calculate horizontal standard deviation of the concentration distribution% following Eckman
    end = len(fc[fc <= cutout])
    y = x[0:int(np.floor(end / 4))]
    xy = x[fc <= cutout]
    fp_xy = np.zeros((len(y), len(xy)))
    for i in range(end):
        for j in range(len(y)):
            # Calculate Eckman's sigma_y based on the standard deviation in the lateral wind fluctuations
            sigma_y = calc_sigma_y(zo, sigma_v, ustar, x[i])
            # Calculate the lateral diffusion (Eq B3 in Detto et al. 2005)
            d_y = np.exp(-0.5 * (y[j] * (1. / sigma_y)) ** 2) * (1.0 / (np.sqrt(2.0 * np.pi) * sigma_y))
            # Calculate the 2D source area function (Eq B2 in Detto et al. 2005)
            fp_xy[j, i] = d_y * fp[i]

    # Get the "other (simetrical) half of the footprint
    fp_xy = np.vstack((np.flipud(fp_xy), fp_xy))
    return fp_xy, xy


def footprint_Kljun(ustar, h, le, t_a, zm, zo, sigma_v=None, bl_height=3000,
                    cutout=0.9, n_bins=N_BINS):
    """
    Adaoted from https://footprint.kljun.net/

    Returns
    -------

    References:
    -----------
    .. [Kljun2015]  Kljun, N., Calanca, P., Rotach, M. W., and Schmid, H. P.:
        A simple two-dimensional parameterisation for Flux Footprint Prediction
        (FFP)
        Geosci. Model Dev., 8, 3695–3713, 2015
        https://doi.org/10.5194/gmd-8-3695-2015
    """

    def get_contour_levels(f, dx, dy, rs=None):
        '''Contour levels of f at percentages of f-integral given by rs'''

        # Check input and resolve to default levels in needed
        if not isinstance(rs, (int, float, list)):
            rs = list(np.linspace(0.10, 0.90, 9))
        if isinstance(rs, (int, float)): rs = [rs]

        # Levels
        pclevs = np.empty(len(rs))
        pclevs[:] = np.nan
        ars = np.empty(len(rs))
        ars[:] = np.nan

        sf = np.sort(f, axis=None)[::-1]
        # Masked array for handling potential nan
        msf = ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))

        csf = msf.cumsum().filled(np.nan) * dx * dy
        for ix, r in enumerate(rs):
            dcsf = np.abs(csf - r)
            pclevs[ix] = sf[np.nanargmin(dcsf)]
            ars[ix] = csf[np.nanargmin(dcsf)]

        return [(round(r, 3), ar, pclev)
                for r, ar, pclev in zip(rs, ars, pclevs)]


    def get_contour_vertices(x, y, f, lev):

        cs = plt.contour(x, y, f, [lev])
        plt.close()
        segs = cs.allsegs[0][0]
        xr = [vert[0] for vert in segs]
        yr = [vert[1] for vert in segs]
        # Set contour to None if it's found to reach the physical domain
        if x.min() >= min(segs[:, 0]) or max(segs[:, 0]) >= x.max() or \
                y.min() >= min(segs[:, 1]) or max(segs[:, 1]) >= y.max():
            return [None, None]

        # x,y coords of contour points.
        return [xr, yr]

    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    xstar_end = 30
    oln = 5000  # limit to L for neutral scaling
    t_a_k = t_a + 273.15  # Convert oC to K
    rho = 1.3079 - 0.0045 * t_a  # Density of air (kg/m3)
    ol = mo.calc_L(ustar, t_a_k, rho, CP, h, le)  # Obukhov Length (m)
    if not sigma_v:
        sigma_v = calc_sigma_v(ustar, ol)
    # ===========================================================================
    # Scaled X* for crosswind integrated footprint
    xstar_ci_param = np.linspace(d, xstar_end, n_bins + 2)
    xstar_ci_param = xstar_ci_param[1:]

    # Crosswind integrated scaled F*
    fstar_ci_param = a * (xstar_ci_param - d) ** b * np.exp(-c / (xstar_ci_param - d))
    ind_notnan = ~np.isnan(fstar_ci_param)
    fstar_ci_param = fstar_ci_param[ind_notnan]
    xstar_ci_param = xstar_ci_param[ind_notnan]

    # Scaled sig_y*
    sigystar_param = ac * np.sqrt(bc * xstar_ci_param ** 2 / (1 + cc * xstar_ci_param))

    # ===========================================================================
    # Real scale x and f_ci
    if ol <= 0 or ol >= oln:
        xx = (1 - 19.0 * zm / ol) ** 0.25
        psi_f = np.log((1 + xx ** 2) / 2.) + 2. * np.log((1 + xx) / 2.) \
                - 2. * np.arctan(xx) + np.pi / 2
    else:
        psi_f = -5.3 * zm / ol

    x = xstar_ci_param * zm / (1. - (zm / bl_height)) * (np.log(zm / zo) - psi_f)
    if np.log(zm / zo) - psi_f > 0:
        x_ci = x
        f_ci = fstar_ci_param / zm * (1. - (zm / bl_height)) / (np.log(zm / zo) - psi_f)
    else:
        return None, None

    # Real scale sig_y
    if abs(ol) > oln:
        ol = -1E6
    if ol <= 0:  # convective
        scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.80
    else:  # stable
        scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0
    sigy = sigystar_param / scale_const * zm * sigma_v / ustar
    sigy[sigy < 0] = np.nan

    # Real scale f(x,y)
    dx = np.abs(x_ci[2] - x_ci[1])
    if dx == 0:
        print("Could not compute footprint")
        return None, None

    y_pos = np.arange(0, (len(x_ci) / 2.) * dx * 1.5, dx)
    # f_pos = np.full((len(f_ci), len(y_pos)), np.nan)
    f_pos = np.empty((len(f_ci), len(y_pos)))
    f_pos[:] = np.nan
    for ix in range(len(f_ci)):
        f_pos[ix, :] = f_ci[ix] * 1 / (np.sqrt(2 * np.pi) * sigy[ix]) \
                       * np.exp(-y_pos ** 2 / (2 * sigy[ix] ** 2))

    #Complete footprint for negative y (symmetrical)
    y_neg = - np.fliplr(y_pos[None, :])[0]
    f_neg = np.fliplr(f_pos)
    y = np.concatenate((y_neg[0:-1], y_pos))
    f = np.concatenate((f_neg[:, :-1].T, f_pos.T)).T

    #Matrices for output
    x_2d = np.tile(x[:,None], (1,len(y)))
    y_2d = np.tile(y.T,(len(x),1))
    f_2d = f
    if np.size(f_2d) == 0:
        return None, None

    # Crop domain and footprint to the cutout value
    dy = dx
    clevs = get_contour_levels(f_2d, dx, dy, cutout)
    xrs, yrs = get_contour_vertices(x_2d, y_2d, f_2d, clevs[0][2])
    if not isinstance(xrs, type(None)) and not isinstance(yrs, type(None)) :
        xrs_crop = [x for x in xrs if x is not None]
        yrs_crop = [x for x in yrs if x is not None]
        dminx = np.floor(min(xrs_crop))
        dmaxx = np.ceil(max(xrs_crop))
        dminy = np.floor(min(yrs_crop))
        dmaxy = np.ceil(max(yrs_crop))
        jrange = np.where((y_2d[0] >= dminy) & (y_2d[0] <= dmaxy))[0]
        jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
        jrange = jrange[np.where((jrange >= 0) & (jrange <= y_2d.shape[0] - 1))[0]]
        irange = np.where((x_2d[:, 0] >= dminx) & (x_2d[:, 0] <= dmaxx))[0]
        irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
        irange = irange[np.where((irange >= 0) & (irange <= x_2d.shape[1] - 1))[0]]
        jrange = [[it] for it in jrange]
        x_2d = x_2d[irange, jrange]
        f_2d = f_2d[irange, jrange]
    return f_2d, x_2d[0]



def footprint_Hsieh(ustar, h, le, t_a, zm, zo, n_bins=N_BINS):
    '''Estimates the footprint (or source weight function) along the upwind distance
        (x=0 is measuring point)

    Authors:    Gaby Katul and Cheng-I Hsieh

    Parameters
    ----------
    ustar : friction velocity (m/s)
    H     : Sensible heat (W/m2)
    LE     : Latent heat (W/m2)
    Ta    : Mean air temperature (oC)
    zm    : Instrument height (m)
    zo    : Momentum roughness height (m)

    Returns
    -------
    Fc : Cumulative source contribution with distance (fraction)
    fp : Source-weight function (with distance)
    L  : Obukhov length (m)
    xp : Peak distance from measuring point to the maximum contributing source area (m)
    x  : Distance vector used in the computation of Fc, fp (m)
    F2H: Fetch to Height ratio (for 90% of flux recovery, i.e. 100:1 or 20:1)

    References:
    -----------
    .. [Hsieh2000] Hsieh, C.I., G.G. Katul, and T.W. Chi, 2000,
        "An approximate analytical model for footprint estimation of scalar fluxes
        in thermally stratified atmospheric flows",
        Advances in Water Resources, 23, 765-772.
    '''

    t_a_k = t_a + 273.15  # Convert oC to K
    rho = 1.3079 - 0.0045 * t_a  # Density of air (kg/m3)
    L = mo.calc_L(ustar, t_a_k, rho, CP, h, le)  # Obukhov Length (m)
    # Get the constant as function of stability
    zu = zm * (np.log(zm / zo) - 1.0 + zo / zm)
    P = [0.59, 1.00, 1.33]
    D = [0.28, 0.97, 2.44]
    Mu = [1000.0, 1000.0, 2000.0]
    stab = zu / L
    thresh = 0.04
    ii = 2
    if stab < -thresh:
        ii = 0
    elif abs(stab) < thresh:
        ii = 1
    elif stab > thresh:
        ii = 2

    d1 = D[ii]
    p1 = P[ii]
    mu1 = Mu[ii]
    min_x = (mu1 / 100.0) * zo
    max_x = mu1 * zm
    x_bin = (max_x - min_x) / n_bins
    x = np.arange(min_x, max_x, x_bin)
    # Eq 16 in Hsieh et al. 2000
    c1 = (-1.0 / (x * KARMAN ** 2)) * (d1 * (zu ** p1) * abs(L) ** (1.0 - p1))
    fc = np.exp(c1)
    # 1D footrprint (Eq 17 in Hiseh et al. 2000)
    fp = -(c1 / x) * fc
    # Peak location of the footrpint (Eq. 19 in Hsieh et al. 2000)
    xp = (1. / (2.0 * KARMAN ** 2)) * (d1 * zu ** p1 * abs(L) ** (1.0 - p1))
    # Fetch to height ratio (Eq. 20 in Hsieh et al. 2000)
    f2_h = (d1 / (0.105 * KARMAN ** 2)) * (zm ** (-1.0) * abs(L) ** (1.00 - p1) * zu ** (p1))
    return fc, fp, x, L, xp, f2_h


def calc_sigma_y(zo, sigma_v, ustar, x):
    '''Estimate horizontal standard deviation of the concentration distribution

    Parameters
    ----------
    zo          : momentum roughness height (m)
    sigma_v     : standard deviation of the horizontal crosswind velocity fluctuations
    ustar       : friction velocity (m/s)
    x           : distance vector into the upwind direction from measurement point (m)

    Returns
    -------
    sigma_y = horizontal standard deviation of the concentration distribution

    following Eckman, R., 1994, "Re-examination of empirically derived formulas
        for horizontal diffusion from surface sources", Atmospheric Environment,
        28, 265-272'''

    a = 0.3  # average value of constant from table 1 of Eckman, 1994
    p = 0.85  # average value of constant from table 1 of Eckman, 1994
    # Equation 17 from Eckman, 1994
    sigma_y = a * zo * sigma_v / ustar * (x / zo) ** p
    return sigma_y


def calc_sigma_v(ustar, l_mo, bl_height=3000.):
    '''% estimate sigma_v using bounary layer height = 3000m

    Parameters
    ----------
    l_mo : Obukhov length (m)
    ustar : friction velocity (m s-1)

    Returns
    -------
    sigma_v : standard deviation of the cross wind velocity'''
    if l_mo > 0:
        l_mo = np.inf
    sigma_v = ustar * (12.0 - 0.5 * bl_height / l_mo) ** (1.0 / 3.0)
    return sigma_v


def geolocate_2d_footprint(fp_xy, xy, windir, tower_position, projection, outputfile):
    ''' Saves footprint Fp_xy in a georreferenced GDAL file

    Parameters
    ----------
    fp_xy :
    xy : Distance vector used in the computation of Fc, fp (m)
    windir : wind direction (degrees)
    tower_position : (X,Y) coordinates in the projected system
    projection: OGR projection system object
    outputfile : path to the output image file
    '''

    # first shift the origin of the footprint to the centre of the matrix
    rows, cols = np.shape(fp_xy)
    temp = np.zeros((rows * 2, cols * 2))
    rows, cols = np.shape(temp)
    temp[int(rows / 4):int(3 * rows / 4), int(cols / 2):cols] = fp_xy
    fp_xy = np.array(temp)
    # Then rotate the matrx
    # since the footprint model is oriented eastwards the windir azimuth angle
    # (0=north, 90 East) must be substracted 90
    fp_xy = rotate(fp_xy, -(windir - 90.0), reshape=True, order=1)
    # Normalize the footprint contributions
    fp_xy = fp_xy / np.sum(fp_xy)
    # locate tower at the centre of the footprint image and geolocate the image
    # GDAL geolocation model considers the top left pixel, and real world pixel
    # size must agree with the axis directions
    rows, cols = np.shape(fp_xy)
    delta_x = np.nanmean(xy[2:] - xy[1:-1])
    if delta_x == 0:
        print("Could not compute geotransform")
        return False

    delta_y = -delta_x
    ul = np.array(tower_position) - np.array([int(delta_x * cols / 2.0), int(delta_y * rows / 2.0)])
    geolocation = [ul[0], delta_x, 0, ul[1], 0, delta_y]
    # Save output TIFF file
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(outputfile, cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(geolocation)
    ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(fp_xy)
    del ds
    if exists(outputfile):
        return True
    else:
        return False


def resample_footprint(input_file, out_gt, out_size, out_proj):
    '''Resamples a 2D footprint image file to a desired extent expressed by rows,cols & Geolocation

    Parameters
    ----------
    outputfile : path to a GDAL image with the footprint in band 1
    outputsize : Size in pixel of the output image (rows,cols)
    gt : GDAL geotransforrm array
        [UL_X,pixel_size_X,Rotation_X,UL_Y,Rotation_Y,pixel_size_Y]

    Returns
    -------
    fp2output : resampled 2d footprint array'''
    # xmin ymax xmax ymin
    extent = [out_gt[0],
              out_gt[3] + out_gt[5] * out_size[0],
              out_gt[0] + out_gt[1] * out_size[1],
              out_gt[3]]

    warp_opts = {"outputBounds": extent,
                 "width": out_size[1],
                 "height": out_size[0],
                 "dstSRS": out_proj,
                 "resampleAlg": "average"}

    tmp_file = input_file.parent / "MEM.tif"
    gdal.Warp(str(tmp_file), str(input_file), **warp_opts)
    fid = gdal.Open(str(tmp_file))
    fp2dout = fid.GetRasterBand(1).ReadAsArray()
    # Normalize the footporint
    fp2dout = fp2dout / np.nansum(fp2dout)
    tmp_file.unlink()
    return fp2dout