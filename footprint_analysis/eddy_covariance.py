from osgeo import gdal
from os.path import exists
import numpy as np
from scipy.ndimage.interpolation import rotate
from pyTSEB import meteo_utils as met
from pyTSEB import MO_similarity as mo

# Global constants
# von Karman's constant
KARMAN = 0.4
N_BINS = 1e3


def footprint_Detto(ustar, h, le, t_a, zm, zo, sigma_v=None, cutout=0.9, n_bins=N_BINS):
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
    [fc, fp, x, *_] = footprint_Hsieh(ustar, h, le, t_a, zm, zo, n_bins=n_bins)
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
    
    Reference:    Hsieh, C.I., G.G. Katul, and T.W. Chi, 2000,  
        "An approximate analytical model for footprint estimation of scalar fluxes 
        in thermally stratified atmospheric flows", 
        Advances in Water Resources, 23, 765-772.'''

    CP = 1005  # Specific heat capacity of dry air at
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


def calc_sigma_v(ustar, l_mo):
    '''% estimate sigma_v using bounary layer height = 3000m
    
    Parameters
    ----------
    l_mo : Obukhov length (m)
    ustar : friction velocity (m s-1)
    
    Returns
    -------
    sigma_v : standard deviation of the cross wind velocity'''

    sigma_v = ustar * (12.0 - 0.5 * 3000.0 / l_mo) ** (1.0 / 3.0)
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
    temp[int(rows / 4):3 * int(rows / 4), int(cols / 2):cols] = fp_xy
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
    delta_x = xy[2] - xy[1]
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


def resample_footprint(outputfile, outputsize, gt):
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

    # open footprint image
    if not exists(outputfile):
        print(outputfile + ' not found')
        return False

    fid = gdal.Open(outputfile, gdal.GA_ReadOnly)
    in_geolocation = fid.GetGeoTransform()
    fp2din = fid.GetRasterBand(1).ReadAsArray()
    inputsize = fp2din.shape
    del fid

    # Create the ouput array
    fp2dout = np.zeros(outputsize)

    # Get the output & input coordinates
    pixels = ((row, col) for row in np.arange(outputsize[0]) for col in np.arange(outputsize[1]))

    incols, inrows = np.meshgrid(np.arange(inputsize[1]), np.arange(inputsize[0]))
    X_in = in_geolocation[0] + in_geolocation[1] * incols + gt[2] * inrows
    Y_in = in_geolocation[3] + in_geolocation[4] * incols + gt[5] * inrows
    # loop the output coordinates to get the total footprint contribution per pixel
    row0 = 0
    for outrow, outcol in pixels:
        if outrow != row0:
            row0 = outrow
            print('Processing row %s' % row0)
        x_out = gt[0] + gt[1] * outcol + gt[2] * outrow
        y_out = gt[3] + gt[4] * outcol + gt[5] * outrow
        x_out1 = x_out + gt[1]
        y_out1 = y_out + gt[5]
        # Find the pixels in the input footprint array that falls in the coordinate grid
        index = np.where(np.logical_and(np.logical_and(X_in >= x_out, X_in <= x_out1),
                                        np.logical_and(Y_in >= y_out1, Y_in <= y_out)))
        if np.size(index) > 0:
            fp_pixel = np.sum(fp2din[index])
            fp2dout[outrow, outcol] = fp_pixel

    # Normalize the footporint
    fp2dout = fp2dout / np.sum(fp2dout)
    return fp2dout
