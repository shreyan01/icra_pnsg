"""
Simulate a stream of PointAndAsk confirmations arriving over an interaction
session, using REAL fitted superquadric params (not hand-typed numbers) so
this test exercises the actual fitter -> registry pipeline end to end.

Scenario: several "short mug" observations, then several "tall mug"
observations (a genuine within-class variant), a "bottle" (new noun), and
one incorrect candidate (F=0) that must NOT affect statistics.
"""
import numpy as np
from superquadric import fit_superquadric, sample_superquadric_surface, add_noise
from registry import Registry

SHORT_MUG_GT = {'a1': 0.04, 'a2': 0.04, 'a3': 0.045, 'eps1': 0.3, 'eps2': 1.0,
                'cx': 0, 'cy': 0, 'cz': 0.045, 'roll': 0, 'pitch': 0, 'yaw': 0}
TALL_MUG_GT = {'a1': 0.035, 'a2': 0.035, 'a3': 0.09, 'eps1': 0.3, 'eps2': 1.0,
               'cx': 0, 'cy': 0, 'cz': 0.09, 'roll': 0, 'pitch': 0, 'yaw': 0}
BOTTLE_GT = {'a1': 0.03, 'a2': 0.03, 'a3': 0.12, 'eps1': 0.4, 'eps2': 1.0,
             'cx': 0, 'cy': 0, 'cz': 0.12, 'roll': 0, 'pitch': 0, 'yaw': 0}


def observe(gt_params, seed):
    """Simulate one noisy, partial-view observation and fit it -- this is
    what a real perception frame would hand to the registry."""
    rng = np.random.default_rng(seed)
    cloud = sample_superquadric_surface(gt_params, n_points=2500, rng=rng)
    # crop to a partial view with slightly randomized camera direction,
    # simulating natural viewpoint variation across interactions
    cam_dir = np.array([0.3, -0.2, 1.0]) + rng.normal(0, 0.05, 3)
    from superquadric import crop_partial_view
    cloud = crop_partial_view(cloud, camera_dir=cam_dir, keep_fraction=0.55)
    cloud = add_noise(cloud, sigma=0.0025, rng=rng)
    fitted, info = fit_superquadric(cloud)
    return fitted, info


def main():
    reg = Registry()
    seed = 100

    print('=== Session: confirming several short mugs ===')
    for i in range(6):
        fitted, info = observe(SHORT_MUG_GT, seed); seed += 1
        entry = reg.confirm(fitted, 'mug', F=1, crop_ref=f'crop_{seed}.png')
        print(f'  obs {i}: rmse={info["rmse"]:.4f}  -> {entry["action"]} ({entry.get("mode_id")})')

    print('\n=== Registry self-report after short mugs ===')
    print(reg.describe('mug'))

    print('\n=== Session: one INCORRECT candidate (F=0) -- should not update stats ===')
    fitted, info = observe(TALL_MUG_GT, seed); seed += 1
    entry = reg.confirm(fitted, 'mug', F=0)
    print(f'  -> {entry["action"]} (statistics untouched)')

    print('\n=== Session: confirming several tall mugs (genuine variant) ===')
    for i in range(6):
        fitted, info = observe(TALL_MUG_GT, seed); seed += 1
        entry = reg.confirm(fitted, 'mug', F=1, crop_ref=f'crop_{seed}.png')
        print(f'  obs {i}: rmse={info["rmse"]:.4f}  -> {entry["action"]} ({entry.get("mode_id")})'
              f'  mahal={entry.get("mahalanobis", 0):.2f}')

    print('\n=== Registry self-report after tall mugs added ===')
    print(reg.describe('mug'))

    print('\n=== Session: confirming a brand-new word, "bottle" ===')
    for i in range(3):
        fitted, info = observe(BOTTLE_GT, seed); seed += 1
        entry = reg.confirm(fitted, 'bottle', F=1, crop_ref=f'crop_{seed}.png')
        print(f'  obs {i}: -> {entry["action"]} ({entry.get("mode_id")})')

    print('\n=== Registry self-report: bottle ===')
    print(reg.describe('bottle'))

    print('\n=== Matching test: does a NEW short-mug observation correctly match the short-mug mode? ===')
    fitted, info = observe(SHORT_MUG_GT, seed); seed += 1
    mu, mode_id = reg.match(fitted, 'mug')
    print(f'  membership={mu:.3f}  matched_mode={mode_id}')

    print('\n=== Provenance check: trace one mode back to its confirming observations ===')
    mug_modes = reg.modes['mug']
    short_mode = mug_modes[0]
    trail = reg.provenance_for_mode(short_mode.mode_id)
    print(f'  mode {short_mode.mode_id} has {len(trail)} confirming observation(s) on record')

    reg.save('/home/claude/pnsg/src/registry_test_output.json')
    print('\nSaved registry_test_output.json')


if __name__ == '__main__':
    main()
