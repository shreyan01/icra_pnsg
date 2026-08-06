"""
Compositional structure test: a synthetic 'mug' = body (cylinder-ish
superquadric) + handle (small elongated superquadric, offset to the side).

IMPORTANT MODELING CAVEAT (documented, not hidden): a superquadric is
always genus-0 (topologically a sphere) and can never represent a hole.
A real mug handle is genus-1 (a loop). The 'handle' superquadric fitted
here is therefore a solid blob approximating the handle's rough position,
size, and elongation -- not its ring topology. This is sufficient for
(a) detecting that a handle-like protrusion exists and where, and
(b) computing a grasp point on it, but NOT for reconstructing the hole
itself. This limitation should be stated explicitly in the paper.

This script assumes ground-truth part segmentation (as if a perfect
segmenter handed us body points and handle points separately) -- the
goal here is to test whether PART-WISE FITTING + A RELATION produces a
useful compositional, human-readable description, not to test
segmentation itself.
"""
import numpy as np
from superquadric import (
    fit_superquadric, sample_superquadric_surface,
    add_noise, local_to_world,
)

# --- Ground truth part definitions ------------------------------------
BODY_GT = {
    'a1': 0.04, 'a2': 0.04, 'a3': 0.06,
    'eps1': 0.3, 'eps2': 1.0,
    'cx': 0.0, 'cy': 0.0, 'cz': 0.06,
    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
}
# Handle: thin, elongated blob attached to the +x side of the body,
# roughly vertical, bent is NOT modeled (flat approximation only).
HANDLE_GT = {
    'a1': 0.012, 'a2': 0.03, 'a3': 0.028,
    'eps1': 0.6, 'eps2': 0.8,
    'cx': 0.055, 'cy': 0.0, 'cz': 0.06,   # offset outward from body center, side-attached
    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
}


def describe_relation(body_params, handle_params):
    """Compute a simple, human-readable spatial relation between two fitted
    parts: offset direction/distance from body center, and whether the
    handle sits within the body's vertical extent (a crude 'attached to
    the side, not the top/bottom' check)."""
    offset = np.array([handle_params['cx'], handle_params['cy'], handle_params['cz']]) - \
             np.array([body_params['cx'], body_params['cy'], body_params['cz']])
    dist = np.linalg.norm(offset)
    horiz_dist = np.linalg.norm(offset[:2])
    vert_offset = offset[2]
    body_half_height = body_params['a3']

    if horiz_dist > 0.5 * max(body_params['a1'], body_params['a2']) and \
       abs(vert_offset) < body_half_height:
        side = 'side (lateral protrusion, mid-height)'
    elif vert_offset > 0.7 * body_half_height:
        side = 'top'
    elif vert_offset < -0.7 * body_half_height:
        side = 'bottom'
    else:
        side = 'unclear'

    return {'offset': offset, 'distance': dist, 'attachment': side}


def shape_word(eps1, eps2):
    """Very crude eps -> natural-language shape label, just for the
    explanation string -- not a real classifier."""
    if eps1 < 0.4 and eps2 < 0.4:
        return 'boxy'
    if eps1 < 0.6 and eps2 > 0.7:
        return 'flattened/disc-like'
    if 0.7 <= eps1 <= 1.3 and 0.7 <= eps2 <= 1.3:
        return 'rounded (cylinder/sphere-like)'
    return 'irregular'


def main():
    rng = np.random.default_rng(0)

    body_cloud = sample_superquadric_surface(BODY_GT, n_points=3000, rng=rng)
    handle_cloud = sample_superquadric_surface(HANDLE_GT, n_points=800, rng=rng)

    body_noisy = add_noise(body_cloud, sigma=0.002, rng=rng)
    handle_noisy = add_noise(handle_cloud, sigma=0.002, rng=rng)

    # Fit each part independently (ground-truth segmentation assumed)
    body_fit, body_info = fit_superquadric(body_noisy)
    handle_fit, handle_info = fit_superquadric(handle_noisy)

    print('=== Part-wise fit quality ===')
    print(f'body   rmse={body_info["rmse"]:.5f}  eps1={body_fit["eps1"]:.2f} eps2={body_fit["eps2"]:.2f}  '
          f'size=({body_fit["a1"]:.3f},{body_fit["a2"]:.3f},{body_fit["a3"]:.3f})')
    print(f'handle rmse={handle_info["rmse"]:.5f}  eps1={handle_fit["eps1"]:.2f} eps2={handle_fit["eps2"]:.2f}  '
          f'size=({handle_fit["a1"]:.3f},{handle_fit["a2"]:.3f},{handle_fit["a3"]:.3f})')

    relation = describe_relation(body_fit, handle_fit)

    print('\n=== Composed structural description ===')
    print(f'body:   {shape_word(body_fit["eps1"], body_fit["eps2"])}, '
          f'height={2*body_fit["a3"]*1000:.0f}mm, diameter~{2*body_fit["a1"]*1000:.0f}mm')
    print(f'handle: {shape_word(handle_fit["eps1"], handle_fit["eps2"])}, '
          f'attached to the {relation["attachment"]} of the body, '
          f'{relation["distance"]*1000:.0f}mm from body center')

    print('\n=== Auto-generated explanation sentence ===')
    print(f'"This object has a {shape_word(body_fit["eps1"], body_fit["eps2"])} body '
          f'({2*body_fit["a3"]*1000:.0f}mm tall) with a small protrusion attached to its '
          f'{relation["attachment"]} -- consistent with the learned \'mug\' prototype '
          f'(body + side handle)."')

    return {
        'body_gt': BODY_GT, 'handle_gt': HANDLE_GT,
        'body_fit': body_fit, 'handle_fit': handle_fit,
        'relation': relation,
    }


if __name__ == '__main__':
    main()
