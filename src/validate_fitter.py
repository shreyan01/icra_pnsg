"""
Sanity-check the superquadric fitter: for several known ground-truth shapes,
sample a full surface, crop to a partial (single-camera-like) view, add
sensor noise, fit, and compare recovered params to ground truth.

This is the step-0 validation to run before any real RGB-D data exists.
"""
import numpy as np
from superquadric import (
    fit_superquadric, sample_superquadric_surface,
    crop_partial_view, add_noise, PARAM_ORDER
)

np.set_printoptions(precision=3, suppress=True)

GROUND_TRUTH_SHAPES = {
    'mug_body (cylinder-ish)': {
        'a1': 0.04, 'a2': 0.04, 'a3': 0.06,
        'eps1': 0.3, 'eps2': 1.0,   # flattish top/bottom (e1 small), round cross-section (e2~1)
        'cx': 0.0, 'cy': 0.0, 'cz': 0.06,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    },
    'box (cube-ish)': {
        'a1': 0.05, 'a2': 0.05, 'a3': 0.05,
        'eps1': 0.1, 'eps2': 0.1,   # boxy in both directions
        'cx': 0.2, 'cy': 0.0, 'cz': 0.05,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.3,
    },
    'bottle (tall cylinder)': {
        'a1': 0.03, 'a2': 0.03, 'a3': 0.10,
        'eps1': 0.4, 'eps2': 1.0,
        'cx': -0.2, 'cy': 0.0, 'cz': 0.10,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    },
    'bowl (squat open-ish)': {
        'a1': 0.06, 'a2': 0.06, 'a3': 0.035,
        'eps1': 0.5, 'eps2': 1.0,
        'cx': 0.0, 'cy': 0.2, 'cz': 0.035,
        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
    },
}


def run_case(name, gt_params, keep_fraction=0.55, noise_sigma=0.002, seed=0):
    rng = np.random.default_rng(seed)
    full_cloud = sample_superquadric_surface(gt_params, n_points=4000, rng=rng)
    partial_cloud = crop_partial_view(full_cloud, camera_dir=np.array([0.3, -0.2, 1.0]),
                                       keep_fraction=keep_fraction)
    noisy_cloud = add_noise(partial_cloud, sigma=noise_sigma, rng=rng)

    fitted, info = fit_superquadric(noisy_cloud, verbose=0)

    print(f'--- {name} ---')
    print(f'  points: full={len(full_cloud)}  partial+noisy={len(noisy_cloud)}')
    print(f'  fit success={info["success"]}  rmse={info["rmse"]:.4f}  nfev={info["nfev"]}')
    print(f'  {"param":6s}  {"truth":>8s}  {"fitted":>8s}  {"abs_err":>8s}')
    for k in PARAM_ORDER:
        truth = gt_params[k]
        fit_v = fitted[k]
        # angles: report separately, size/shape params are the ones that matter most
        err = abs(truth - fit_v)
        print(f'  {k:6s}  {truth:8.3f}  {fit_v:8.3f}  {err:8.3f}')
    print()
    return fitted, info


if __name__ == '__main__':
    results = {}
    for name, gt in GROUND_TRUTH_SHAPES.items():
        results[name] = run_case(name, gt)

    print('=== Summary ===')
    for name, (fitted, info) in results.items():
        print(f'{name:28s}  rmse={info["rmse"]:.4f}  success={info["success"]}')
