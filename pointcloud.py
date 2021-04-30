import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def point_cloud(x, y, z, values, N = 10000, isomin=None, isomax=None):

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    if isomin is None:
        isomin = values.min()
        
    if isomax is None:
        isomax = values.max()

    inds = (values >= isomin) & (values <= isomax)
    data = values[inds]
    data = (data - isomin) / (isomax - isomin)
    density = data / data.sum()

    major = np.floor(N * density).astype(int)
    minor = ((N * density - major) > np.random.rand(*density.shape)).astype(int)
    npoints = major + minor
    points = []
    for n, (indx, indy, indz) in zip(npoints, np.argwhere(inds)):

        if n == 0:
            continue

        for _ in range(n):
            pointx = x[indx] + np.random.triangular(-dx, 0, dx)
            pointy = y[indy] + np.random.triangular(-dy, 0, dy)
            pointz = z[indz] + np.random.triangular(-dz, 0, dz)
            points.append([pointx, pointy, pointz])

    points = np.array(points)
    return points
