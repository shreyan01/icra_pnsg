"""
Diagnose the box case: is the messy wireframe a real fitting problem
(genuinely bad/underconstrained shape) or a mesh-sampling/visualization
artifact at low epsilon? Compare full-view vs partial-view fits, and
inspect the surface mesh sampling directly.
"""
import numpy as np
from superquadric import (
    fit_superquadric, sample_superquadric_surface,
    crop_partial_view, add_noise, inside_outside, _fexp
)
from validate_fitter import GROUND_TRUTH_SHAPES

box_gt = GROUND_TRUTH_SHAPES['box (cube-ish)']
rng = np.random.default_rng(0)

# 1. Full view (no cropping) - upper bound on what's achievable
full_cloud = sample_superquadric_surface(box_gt, n_points=4000, rng=rng)
full_noisy = add_noise(full_cloud, sigma=0.002, rng=rng)
fitted_full, info_full = fit_superquadric(full_noisy)
print('=== FULL VIEW (all 6 faces visible) ===')
print(f'rmse={info_full["rmse"]:.5f}')
for k in ['a1', 'a2', 'a3', 'eps1', 'eps2', 'yaw']:
    print(f'  {k:6s} truth={box_gt[k]:.3f}  fitted={fitted_full[k]:.3f}  err={abs(box_gt[k]-fitted_full[k]):.3f}')

# 2. Partial view (as before) - check how many faces are actually visible
partial_cloud = crop_partial_view(full_cloud, camera_dir=np.array([0.3, -0.2, 1.0]), keep_fraction=0.55)
partial_noisy = add_noise(partial_cloud, sigma=0.002, rng=rng)
fitted_partial, info_partial = fit_superquadric(partial_noisy)
print('\n=== PARTIAL VIEW (single camera) ===')
print(f'points kept: {len(partial_cloud)} / {len(full_cloud)} ({100*len(partial_cloud)/len(full_cloud):.0f}%)')
print(f'rmse={info_partial["rmse"]:.5f}')
for k in ['a1', 'a2', 'a3', 'eps1', 'eps2', 'yaw']:
    print(f'  {k:6s} truth={box_gt[k]:.3f}  fitted={fitted_partial[k]:.3f}  err={abs(box_gt[k]-fitted_partial[k]):.3f}')

# 3. Check surface mesh sampling itself at low epsilon - is _fexp well-behaved?
print('\n=== Surface mesh sampling check (low epsilon numerical behavior) ===')
angles = np.linspace(-np.pi, np.pi, 9)
for e in [0.1, 0.127, 0.5, 1.0]:
    vals = _fexp(np.cos(angles), e)
    print(f'  eps={e:.3f}  cos^e range: [{vals.min():.4f}, {vals.max():.4f}]  any_nan={np.any(np.isnan(vals))}')

# 4. Point-to-surface distance check (independent of the fitting residual metric)
# For each point in partial_noisy, find how far it truly sits from the fitted
# surface by checking F (should be close to 1 for a good fit).
F_vals = inside_outside(partial_noisy, fitted_partial)
print('\n=== Inside-outside F distribution on fitted partial-view surface ===')
print(f'  F: min={F_vals.min():.3f} max={F_vals.max():.3f} mean={F_vals.mean():.3f} (should cluster near 1.0)')
