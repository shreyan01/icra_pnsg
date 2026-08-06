"""
Core superquadric math: implicit inside-outside function, surface sampling,
and closed-form fitting to a (possibly partial) point cloud.

Parameterization (11 DOF):
    a1, a2, a3   : half-dimensions along local x, y, z      (scale)
    eps1, eps2   : shape exponents (vertical, horizontal)   (shape)
    cx, cy, cz   : centroid (translation)
    roll, pitch, yaw : orientation (rotation, Euler ZYX)

eps ~ 0.1  -> boxy / square cross-section
eps ~ 1.0  -> cylindrical / spherical (rounded)
eps ~ 2.0  -> pinched / diamond-like
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


def _fexp(base, exponent):
    """Signed power: sign(base) * |base|^exponent. Keeps optimizer well-behaved
    away from base=0 where raw ** would produce NaNs for negative bases."""
    return np.sign(base) * (np.abs(base) ** exponent)


def world_to_local(points, cx, cy, cz, roll, pitch, yaw):
    """Transform world-frame points into the superquadric's local frame."""
    rot = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    centered = points - np.array([cx, cy, cz])
    local = centered @ rot  # inverse rotation = transpose, applied via right-multiply
    return local


def local_to_world(points, cx, cy, cz, roll, pitch, yaw):
    rot = R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    world = points @ rot.T + np.array([cx, cy, cz])
    return world


def inside_outside(points, params):
    """
    Superquadric implicit (inside-outside) function F(x,y,z).
    F == 1 on the surface, F < 1 inside, F > 1 outside.

    points: (N,3) world-frame points
    params: dict with keys a1,a2,a3,eps1,eps2,cx,cy,cz,roll,pitch,yaw
    """
    x, y, z = world_to_local(points, params['cx'], params['cy'], params['cz'],
                              params['roll'], params['pitch'], params['yaw']).T
    a1, a2, a3 = params['a1'], params['a2'], params['a3']
    e1, e2 = params['eps1'], params['eps2']

    term_xy = _fexp(x / a1, 2.0 / e2) ** 1  # placeholder, computed properly below
    # Standard form: ((x/a1)^(2/e2) + (y/a2)^(2/e2))^(e2/e1) + (z/a3)^(2/e1)
    tx = np.abs(x / a1) ** (2.0 / e2)
    ty = np.abs(y / a2) ** (2.0 / e2)
    inner = (tx + ty) ** (e2 / e1)
    tz = np.abs(z / a3) ** (2.0 / e1)
    F = inner + tz
    return F


def radial_residual(points, params):
    """
    Residual used for fitting: (F^(e1/2) - 1) * sqrt(a1*a2*a3)
    This is the standard Solina-Bajcsy style residual. Raising F to e1/2
    linearizes the residual near the surface (avoids over-penalizing
    points far from flat/boxy shapes), and scaling by shape volume keeps
    residuals comparable across different object sizes.
    """
    F = inside_outside(points, params)
    e1 = params['eps1']
    vol_scale = np.sqrt(params['a1'] * params['a2'] * params['a3'])
    return (F ** (e1 / 2.0) - 1.0) * vol_scale


PARAM_ORDER = ['a1', 'a2', 'a3', 'eps1', 'eps2', 'cx', 'cy', 'cz', 'roll', 'pitch', 'yaw']


def params_to_vec(params):
    return np.array([params[k] for k in PARAM_ORDER])


def vec_to_params(vec):
    return dict(zip(PARAM_ORDER, vec))


def initial_guess(points):
    """Cheap, sane initialization from point cloud statistics: centroid for
    translation, PCA axes ignored for now (assume roughly axis-aligned local
    frame — fine for tabletop objects viewed from above), bounding-box half
    extents for scale, eps=1.0 (cylinder/sphere-like) as a neutral shape prior."""
    centroid = points.mean(axis=0)
    extents = (points.max(axis=0) - points.min(axis=0)) / 2.0
    extents = np.clip(extents, 1e-3, None)
    return {
        'a1': extents[0], 'a2': extents[1], 'a3': extents[2],
        'eps1': 1.0, 'eps2': 1.0,
        'cx': centroid[0], 'cy': centroid[1], 'cz': centroid[2],
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    }


BOUNDS_LOW = {
    'a1': 1e-3, 'a2': 1e-3, 'a3': 1e-3,
    'eps1': 0.1, 'eps2': 0.1,
    'cx': -np.inf, 'cy': -np.inf, 'cz': -np.inf,
    'roll': -np.pi, 'pitch': -np.pi, 'yaw': -np.pi,
}
BOUNDS_HIGH = {
    'a1': np.inf, 'a2': np.inf, 'a3': np.inf,
    'eps1': 1.9, 'eps2': 1.9,
    'cx': np.inf, 'cy': np.inf, 'cz': np.inf,
    'roll': np.pi, 'pitch': np.pi, 'yaw': np.pi,
}


def fit_superquadric(points, init=None, verbose=0):
    """
    Fit a superquadric to a (possibly partial) point cloud via nonlinear
    least squares on the radial residual.

    Returns: (params dict, residual info dict)
    """
    if init is None:
        init = initial_guess(points)

    x0 = params_to_vec(init)
    lo = np.array([BOUNDS_LOW[k] for k in PARAM_ORDER])
    hi = np.array([BOUNDS_HIGH[k] for k in PARAM_ORDER])

    def resid_fn(vec):
        p = vec_to_params(vec)
        return radial_residual(points, p)

    result = least_squares(
        resid_fn, x0, bounds=(lo, hi),
        method='trf', loss='soft_l1', f_scale=0.05,
        max_nfev=3000, verbose=verbose,
    )

    fitted = vec_to_params(result.x)
    final_resid = resid_fn(result.x)
    info = {
        'success': result.success,
        'cost': result.cost,
        'rmse': float(np.sqrt(np.mean(final_resid ** 2))),
        'max_abs_resid': float(np.max(np.abs(final_resid))),
        'nfev': result.nfev,
    }
    return fitted, info


# ---------------------------------------------------------------------------
# Synthetic data generation, for validating the fitter against known ground
# truth before any real sensor data is available.
# ---------------------------------------------------------------------------

def sample_superquadric_surface(params, n_points=2000, rng=None):
    """Sample points approximately uniformly on a superquadric surface using
    the standard spherical-product parameterization, then map to world frame."""
    if rng is None:
        rng = np.random.default_rng(0)

    eta = rng.uniform(-np.pi / 2, np.pi / 2, n_points)   # latitude
    omega = rng.uniform(-np.pi, np.pi, n_points)         # longitude

    e1, e2 = params['eps1'], params['eps2']
    a1, a2, a3 = params['a1'], params['a2'], params['a3']

    def sc(angle, e):
        return _fexp(np.cos(angle), e)

    def ss(angle, e):
        return _fexp(np.sin(angle), e)

    x = a1 * sc(eta, e1) * sc(omega, e2)
    y = a2 * sc(eta, e1) * ss(omega, e2)
    z = a3 * ss(eta, e1)

    local_pts = np.stack([x, y, z], axis=1)
    world_pts = local_to_world(local_pts, params['cx'], params['cy'], params['cz'],
                                params['roll'], params['pitch'], params['yaw'])
    return world_pts


def crop_partial_view(points, camera_dir=np.array([0, 0, 1]), keep_fraction=0.5):
    """Simulate a single-camera partial view: keep only points whose outward
    normal-ish direction (approximated via position relative to centroid)
    faces the camera. Crude but adequate for stress-testing the fitter."""
    centroid = points.mean(axis=0)
    rel = points - centroid
    rel_norm = rel / (np.linalg.norm(rel, axis=1, keepdims=True) + 1e-9)
    dot = rel_norm @ (camera_dir / np.linalg.norm(camera_dir))
    thresh = np.quantile(dot, 1.0 - keep_fraction)
    mask = dot >= thresh
    return points[mask]


def add_noise(points, sigma=0.003, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    return points + rng.normal(0, sigma, points.shape)
