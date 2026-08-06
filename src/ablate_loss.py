"""
Ablation: does the choice of robust loss (and its scale) in least_squares
explain the systematic eps bias (true 0.10 -> fitted ~0.13) seen even in
full-view, low-noise conditions? Test on the box case, full view, so
occlusion is not a confound.
"""
import numpy as np
from scipy.optimize import least_squares
from superquadric import (
    sample_superquadric_surface, add_noise, radial_residual,
    initial_guess, params_to_vec, vec_to_params, PARAM_ORDER,
    BOUNDS_LOW, BOUNDS_HIGH,
)
from validate_fitter import GROUND_TRUTH_SHAPES

box_gt = GROUND_TRUTH_SHAPES['box (cube-ish)']
rng = np.random.default_rng(0)
cloud = sample_superquadric_surface(box_gt, n_points=4000, rng=rng)
cloud = add_noise(cloud, sigma=0.002, rng=np.random.default_rng(1))

init = initial_guess(cloud)
x0 = params_to_vec(init)
lo = np.array([BOUNDS_LOW[k] for k in PARAM_ORDER])
hi = np.array([BOUNDS_HIGH[k] for k in PARAM_ORDER])


def resid_fn(vec):
    p = vec_to_params(vec)
    return radial_residual(cloud, p)


configs = [
    dict(loss='linear', f_scale=1.0),
    dict(loss='soft_l1', f_scale=0.05),   # current default
    dict(loss='soft_l1', f_scale=0.5),
    dict(loss='huber', f_scale=0.05),
    dict(loss='cauchy', f_scale=0.05),
]

print(f'{"loss/f_scale":22s} {"eps1":>7s} {"eps2":>7s} {"a1":>7s} {"rmse":>9s}')
print(f'{"(truth)":22s} {box_gt["eps1"]:7.3f} {box_gt["eps2"]:7.3f} {box_gt["a1"]:7.3f} {"-":>9s}')
for cfg in configs:
    result = least_squares(resid_fn, x0, bounds=(lo, hi), method='trf',
                            max_nfev=3000, **cfg)
    fitted = vec_to_params(result.x)
    final_resid = resid_fn(result.x)
    rmse = np.sqrt(np.mean(final_resid ** 2))
    label = f'{cfg["loss"]}/{cfg["f_scale"]}'
    print(f'{label:22s} {fitted["eps1"]:7.3f} {fitted["eps2"]:7.3f} {fitted["a1"]:7.3f} {rmse:9.5f}')
