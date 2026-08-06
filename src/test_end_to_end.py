"""
End-to-end simulated session. A sequence of "utterances" arrive; each
refers to a real (synthetic) object. A simulated detector gives mu_det
(imperfect: correct-ish but noisy). Ground truth determines the simulated
human feedback F when PointAndAsk fires. We track: trigger rate over time
(should fall as vocabulary is learned), alpha_t trajectory, and whether
the registry converges to sensible per-noun prototypes.
"""
import numpy as np
from superquadric import fit_superquadric, sample_superquadric_surface, crop_partial_view, add_noise
from registry import Registry
from loop import IntrospectiveVocabLoop

RNG = np.random.default_rng(42)

OBJECT_TYPES = {
    'mug': {'a1': 0.04, 'a2': 0.04, 'a3': 0.045, 'eps1': 0.3, 'eps2': 1.0,
            'cx': 0, 'cy': 0, 'cz': 0.045, 'roll': 0, 'pitch': 0, 'yaw': 0},
    'bottle': {'a1': 0.03, 'a2': 0.03, 'a3': 0.12, 'eps1': 0.4, 'eps2': 1.0,
               'cx': 0, 'cy': 0, 'cz': 0.12, 'roll': 0, 'pitch': 0, 'yaw': 0},
    'bowl': {'a1': 0.06, 'a2': 0.06, 'a3': 0.035, 'eps1': 0.5, 'eps2': 1.0,
             'cx': 0, 'cy': 0, 'cz': 0.035, 'roll': 0, 'pitch': 0, 'yaw': 0},
}


def observe(gt_params, seed):
    rng = np.random.default_rng(seed)
    cloud = sample_superquadric_surface(gt_params, n_points=2500, rng=rng)
    cam_dir = np.array([0.3, -0.2, 1.0]) + rng.normal(0, 0.05, 3)
    cloud = crop_partial_view(cloud, camera_dir=cam_dir, keep_fraction=0.55)
    cloud = add_noise(cloud, sigma=0.0025, rng=rng)
    fitted, info = fit_superquadric(cloud)
    return fitted, info


def simulated_detector_confidence(true_label, spoken_noun, rng):
    """Stand-in for a real open-vocab detector: if the spoken noun matches
    the object's true category, confidence is high with noise; otherwise
    low with noise (simulating an imperfect but generally competent
    detector)."""
    if true_label == spoken_noun:
        return float(np.clip(rng.normal(0.85, 0.08), 0.0, 1.0))
    else:
        return float(np.clip(rng.normal(0.25, 0.1), 0.0, 1.0))


def make_feedback_fn(true_label):
    def feedback_fn(spoken_noun, fitted_params):
        # ground truth: correct iff spoken noun equals the object's real category
        return 1 if spoken_noun == true_label else 0
    return feedback_fn


def main():
    reg = Registry()
    loop = IntrospectiveVocabLoop(registry=reg, alpha0=0.5)

    # session: mostly correctly-referred objects (as in normal use), each
    # object type appears repeatedly so we can see the vocabulary converge
    labels = list(OBJECT_TYPES.keys())
    session_plan = []
    for i in range(30):
        true_label = labels[i % len(labels)]
        spoken_noun = true_label  # assume correctly-understood commands for this test
        session_plan.append((true_label, spoken_noun))

    seed = 500
    trigger_flags = []
    for t, (true_label, spoken_noun) in enumerate(session_plan):
        fitted, info = observe(OBJECT_TYPES[true_label], seed); seed += 1
        mu_det = simulated_detector_confidence(true_label, spoken_noun, RNG)
        record = loop.step(fitted, spoken_noun, mu_det, make_feedback_fn(true_label))
        trigger_flags.append(record['triggered'])

        tag = 'ASK ' if record['triggered'] else 'act  '
        print(f't={t:2d} [{spoken_noun:6s}] mu_det={mu_det:.2f} mu_obj={record["mu_obj"]:.2f} '
              f'chi={record["chi"]:.2f} alpha={record["alpha_before"]:.2f} -> {tag} '
              f'({record["registry_action"]})')

    print('\n=== Session summary ===')
    print(loop.summary())

    # trigger rate in first third vs last third -- should fall as vocab is learned
    n = len(trigger_flags)
    first_third = trigger_flags[:n // 3]
    last_third = trigger_flags[-n // 3:]
    print(f'\nTrigger rate, first third: {sum(first_third)}/{len(first_third)} '
          f'= {100*sum(first_third)/len(first_third):.0f}%')
    print(f'Trigger rate, last third:  {sum(last_third)}/{len(last_third)} '
          f'= {100*sum(last_third)/len(last_third):.0f}%')

    print('\n=== Final registry state ===')
    for noun in OBJECT_TYPES:
        print(reg.describe(noun))


if __name__ == '__main__':
    main()
