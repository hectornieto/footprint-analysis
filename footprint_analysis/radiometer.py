import numpy as np
import matplotlib.pyplot as plt
import decimal

def cummulative_contribution(h, x):
    """

    Parameters
    ----------
    h
    x

    Returns
    -------

    https://link.springer.com/article/10.1007/s00704-017-2326-z
    """
    
    cumm_x = x**2 / (x**2 + h**2)
    return cumm_x
    

def circle(r, steps=100):
    xs = np.linspace(0, r, steps)
    # y**2 + x**2 = r**2
    ys = np.sqrt(r**2 - xs**2)
    return xs, ys

def footprint_2d(xs, footprint_1d):
    coords = np.block([-np.flip(xs), 0, xs])
    x_coord, y_coord = np.meshgrid(coords, coords)
    quadrant = np.full(x_coord.shape, np.nan)
    radii = np.sqrt(x_coord ** 2 + y_coord**2)
    for x, f in zip(xs, footprint_1d):
        valid = np.logical_and(np.isnan(quadrant), x >= radii)
        quadrant[valid] = f

    quadrant = quadrant / np.nansum(quadrant)
    return x_coord, y_coord, quadrant


def footprint_1d(h, dx=1, max_x=100):
    dists = np.arange(0, max_x, dx)
    f_0 = 0.0
    footprint = []
    xs = []
    for dist in dists:
        xs.append(dist + dx)
        f_1 = cummulative_contribution(h, dist + dx)
        footprint.append(f_1 - f_0)
        f_0 = f_1

    footprint = np.array(footprint)
    footprint = footprint / np.sum(footprint)
    return np.array(xs), footprint



if __name__ == "__main__":
    h = 10  # sensor height (m)
    dx = 20  # Output resolution in the same units as h
    # One-dimensional footprint
    dists, f1d = footprint_1d(h, dx, max_x=100)
    # Two-dimensionall footprint
    x_coord, y_coord, f2d = footprint_2d(dists, f1d)
    fig, axs = plt.subplots(ncols=2)
    axs[0].set_title("1D footprint")
    axs[0].plot(dists, f1d)
    axs[0].set_xlabel("Horizontal distance")
    axs[0].set_xlabel("Relative footprint contribution")
    axs[1].set_title("2D footprint")
    axs[1].imshow(f2d)
    plt.show()
 
