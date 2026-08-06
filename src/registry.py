"""
The vocabulary registry: multi-mode prototypes over superquadric SHAPE
parameters (not pose), with online mean/variance updates, mode
spawning/merging, and full provenance.

Design decisions, stated explicitly:

1. Feature space excludes pose. Only (a1, a2, a3, eps1, eps2) describe the
   CONCEPT; (cx, cy, cz, roll, pitch, yaw) describe where a particular
   instance happens to sit on the table, which is irrelevant to "what is a
   mug". This matches the paper's explainability principle: every
   dimension in the concept representation must be nameable and concept-
   relevant, not incidental.

2. a1/a2 are canonicalized (sorted so a1 >= a2) before matching/updating,
   since for near-rotationally-symmetric objects the assignment of which
   horizontal axis is "a1" vs "a2" is an artifact of the fitter's initial
   orientation guess, not a real shape difference. This is a known
   simplification -- it assumes objects are roughly upright with a
   vertical axis of interest (a3), which holds for tabletop manipulation
   but would need revisiting for arbitrary orientations.

3. Online statistics use Welford's algorithm (true running mean/variance)
   rather than a fixed-rate exponential update. This is a deliberate
   departure from the original PNSG spec's fixed lambda_vocab=0.05 rule:
   Welford gives a statistically principled variance that shrinks as
   evidence accumulates (a mode with 3 confirmations is honestly reported
   as more uncertain than one with 30), which strengthens the paper's
   "honest uncertainty" claim. The tradeoff: Welford's mean stops
   adapting quickly to *drift* once n is large (each new point matters
   less), whereas a fixed learning rate keeps adapting at a constant
   pace. For now we accept this tradeoff since environments are assumed
   roughly stable within a deployment; a hybrid (e.g., a rate floor) is
   a documented future option, not implemented here.

4. Only F=1 (confirmed correct) observations update statistics, per the
   original design -- this prevents wrong candidates from corrupting the
   prototype. F=0 observations are still logged to provenance for
   auditability, just not used to update the mean/variance.
"""
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

FEATURE_KEYS = ['a1', 'a2', 'eps1', 'eps2', 'a3']  # canonical order, a3 kept separate conceptually
SPAWN_K_SIGMA = 2.5       # Mahalanobis-distance threshold to spawn a new mode
DEFAULT_INIT_STD = np.array([0.01, 0.01, 0.15, 0.15, 0.01])  # prior std before n>=2
MAX_MODES_PER_NOUN = 5
MIN_STD = 1e-4             # floor to avoid divide-by-zero on a mode with n=1
PRIOR_PSEUDO_N = 4          # shrinkage strength: how many "virtual" prior samples
                             # the DEFAULT_INIT_STD prior is worth. With few real
                             # observations, std stays close to the prior instead
                             # of collapsing toward zero by chance; as n grows,
                             # the empirical variance dominates.


def canonicalize(params: dict) -> np.ndarray:
    """Extract the concept-relevant feature vector from a fitted superquadric,
    with a1/a2 sorted so the horizontal-axis labeling ambiguity doesn't
    create spurious mode splits."""
    a1, a2 = sorted([params['a1'], params['a2']], reverse=True)
    return np.array([a1, a2, params['eps1'], params['eps2'], params['a3']])


@dataclass
class Mode:
    mean: np.ndarray
    m2: np.ndarray            # Welford running sum of squared deviations
    n: int
    mode_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def std(self) -> np.ndarray:
        """Shrinkage-blended std: pools the empirical Welford variance with
        the DEFAULT_INIT_STD prior, weighted by PRIOR_PSEUDO_N virtual
        samples. This prevents a mode with only 1-2 real observations from
        reporting a spuriously tiny variance (which would falsely trigger
        mode-spawning on the very next, genuinely-similar observation) --
        the empirical variance dominates only once n comfortably exceeds
        the prior's pseudo-count."""
        prior_var = DEFAULT_INIT_STD ** 2
        if self.n < 2:
            emp_var = prior_var.copy()
            emp_n = 0
        else:
            emp_var = self.m2 / (self.n - 1)
            emp_n = self.n - 1
        blended_var = (PRIOR_PSEUDO_N * prior_var + emp_n * emp_var) / (PRIOR_PSEUDO_N + emp_n)
        return np.maximum(np.sqrt(blended_var), MIN_STD)

    def mahalanobis(self, f: np.ndarray) -> float:
        d = (f - self.mean) / self.std
        return float(np.sqrt(np.sum(d ** 2)))

    def membership(self, f: np.ndarray) -> float:
        """Gaussian membership using per-dimension std -- this IS mu_obj
        for this mode."""
        d = (f - self.mean) / self.std
        return float(np.exp(-0.5 * np.sum(d ** 2)))

    def update(self, f: np.ndarray):
        """Welford online update."""
        self.n += 1
        delta = f - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = f - self.mean
        self.m2 = self.m2 + delta * delta2

    def to_dict(self):
        return {'mode_id': self.mode_id, 'mean': self.mean.tolist(),
                'm2': self.m2.tolist(), 'n': self.n}

    @staticmethod
    def from_dict(d):
        return Mode(mean=np.array(d['mean']), m2=np.array(d['m2']),
                    n=d['n'], mode_id=d['mode_id'])

    @staticmethod
    def bootstrap(f: np.ndarray):
        return Mode(mean=f.copy(), m2=np.zeros_like(f), n=1)


class Registry:
    def __init__(self):
        self.modes: dict[str, list[Mode]] = {}
        self.provenance: list[dict] = []   # append-only log

    # -- matching -----------------------------------------------------
    def match(self, params: dict, noun: str) -> tuple[Optional[float], Optional[str]]:
        """Return (mu_obj, mode_id) for the best-matching mode of `noun`,
        or (None, None) if the noun is entirely unknown (triggers the
        unknown-object fallback upstream, per the original design)."""
        if noun not in self.modes or not self.modes[noun]:
            return None, None
        f = canonicalize(params)
        scored = [(m.membership(f), m.mode_id) for m in self.modes[noun]]
        return max(scored, key=lambda x: x[0])

    # -- learning -------------------------------------------------------
    def confirm(self, params: dict, noun: str, F: int, crop_ref: str = None) -> dict:
        """Process a PointAndAsk confirmation. Always logs to provenance;
        only F=1 updates the registry's statistics."""
        f = canonicalize(params)
        entry = {
            'ts': time.time(), 'noun': noun, 'F': F,
            'features': f.tolist(), 'crop_ref': crop_ref,
        }

        if F != 1:
            entry['action'] = 'logged_only_incorrect'
            self.provenance.append(entry)
            return entry

        modes = self.modes.setdefault(noun, [])

        if not modes:
            new_mode = Mode.bootstrap(f)
            modes.append(new_mode)
            entry['action'] = 'bootstrapped_new_noun'
            entry['mode_id'] = new_mode.mode_id
            self.provenance.append(entry)
            return entry

        dists = [(m.mahalanobis(f), m) for m in modes]
        best_dist, best_mode = min(dists, key=lambda x: x[0])

        if best_dist <= SPAWN_K_SIGMA:
            best_mode.update(f)
            entry['action'] = 'updated_existing_mode'
            entry['mode_id'] = best_mode.mode_id
            entry['mahalanobis'] = best_dist
        else:
            new_mode = Mode.bootstrap(f)
            modes.append(new_mode)
            entry['action'] = 'spawned_new_mode'
            entry['mode_id'] = new_mode.mode_id
            entry['mahalanobis'] = best_dist
            self._maybe_merge(noun)

        self.provenance.append(entry)
        return entry

    def _maybe_merge(self, noun: str):
        """If a noun exceeds the mode cap, merge the two closest modes
        (n-weighted mean, pooled variance) to keep memory bounded."""
        modes = self.modes[noun]
        if len(modes) <= MAX_MODES_PER_NOUN:
            return

        best_pair, best_dist = None, np.inf
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                d = np.linalg.norm(modes[i].mean - modes[j].mean)
                if d < best_dist:
                    best_dist, best_pair = d, (i, j)

        i, j = best_pair
        a, b = modes[i], modes[j]
        n_total = a.n + b.n
        merged_mean = (a.mean * a.n + b.mean * b.n) / n_total
        # pooled variance (combine within-group + between-group terms)
        var_a = a.m2 / max(a.n - 1, 1)
        var_b = b.m2 / max(b.n - 1, 1)
        pooled_var = ((a.n - 1) * var_a + (b.n - 1) * var_b) / max(n_total - 2, 1)
        pooled_var += (a.n * b.n / n_total) * ((a.mean - b.mean) ** 2) / n_total
        merged = Mode(mean=merged_mean, m2=pooled_var * max(n_total - 1, 1), n=n_total)

        new_modes = [m for k, m in enumerate(modes) if k not in (i, j)]
        new_modes.append(merged)
        self.modes[noun] = new_modes

    # -- introspection / reporting --------------------------------------
    def describe(self, noun: str) -> str:
        """Human-readable self-report of what the registry has learned
        about a word -- the 'robot reports its own ontology' capability."""
        if noun not in self.modes or not self.modes[noun]:
            return f'I have no learned prototype for "{noun}" yet.'
        lines = [f'I know {len(self.modes[noun])} variant(s) of "{noun}":']
        for m in self.modes[noun]:
            a1, a2, e1, e2, a3 = m.mean
            lines.append(
                f'  - variant {m.mode_id} (n={m.n}): '
                f'~{a1*2000:.0f}x{a2*2000:.0f}mm footprint, {a3*2000:.0f}mm tall, '
                f'shape exponents ({e1:.2f}, {e2:.2f})'
            )
        return '\n'.join(lines)

    def provenance_for_mode(self, mode_id: str) -> list[dict]:
        return [e for e in self.provenance if e.get('mode_id') == mode_id]

    # -- persistence ------------------------------------------------------
    def save(self, path: str):
        data = {
            'modes': {noun: [m.to_dict() for m in modes] for noun, modes in self.modes.items()},
            'provenance': self.provenance,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> 'Registry':
        with open(path) as f:
            data = json.load(f)
        reg = Registry()
        reg.modes = {noun: [Mode.from_dict(d) for d in modes]
                     for noun, modes in data['modes'].items()}
        reg.provenance = data['provenance']
        return reg
