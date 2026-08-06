"""
End-to-end introspective vocabulary loop: wires the superquadric fitter and
the registry together with the two-source conflict gate (chi/alpha_t)
carried over from the RO-MAN 2026 paper.

Per-trial flow:
  1. A point cloud is fitted to a superquadric (the "what does it look
     like" evidence).
  2. The registry scores that shape against the spoken noun's learned
     prototype -> mu_obj (the "does it match what I've learned" evidence).
  3. A (simulated, for now) detector confidence mu_det stands in for the
     "what does it look like it's called" evidence.
  4. chi = (mu_det - mu_obj)^2 is the two-source conflict score: the
     squared disagreement between semantic and geometric evidence. This
     is the closed-form reduction of the RO-MAN normalised-variance
     formula to exactly two sources (Var({p,q}) / max_var = (p-q)^2 for
     n=2), so chi in [0,1] with no additional scaling needed.
  5. If chi > alpha_t: PointAndAsk -> get feedback F -> update the
     registry (vocabulary learning) AND update alpha_t (adaptive
     self-confidence), exactly as in RO-MAN eq. 7-8.
     Else: act autonomously, no learning signal this trial.

Unknown nouns (no registry entry at all) always trigger PointAndAsk
regardless of chi, since there is no mu_obj to compare against yet --
this is how new vocabulary gets bootstrapped.
"""
import numpy as np
from registry import Registry

ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
ETA_BASE = 0.05
M_ASSISTED = 0.3   # PointAndAsk-triggered feedback (only path that updates alpha here)


class IntrospectiveVocabLoop:
    def __init__(self, registry: Registry = None, alpha0: float = 0.5):
        self.registry = registry or Registry()
        self.alpha = alpha0
        self.history = []

    def _update_alpha(self, F: int, chi: float):
        eta = ETA_BASE * M_ASSISTED * (0.5 + self.alpha)
        w = (2 * F - 1) * (1 + chi)
        A = (1 + (1 - self.alpha)) if F == 1 else (1 + self.alpha)
        raw = eta * w * A
        if raw > 0:
            B = (ALPHA_MAX - self.alpha) / (ALPHA_MAX - ALPHA_MIN)
        else:
            B = (self.alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        delta = raw * B
        self.alpha = float(np.clip(self.alpha + delta, ALPHA_MIN, ALPHA_MAX))

    def step(self, fitted_params: dict, noun: str, mu_det: float,
              feedback_fn) -> dict:
        """
        feedback_fn(noun, fitted_params) -> F in {0,1}: stands in for asking
        the human. In simulation this checks against ground truth; on the
        real robot this is the PointAndAsk routine's return value.
        """
        mu_obj, mode_id = self.registry.match(fitted_params, noun)

        unknown = mu_obj is None
        if unknown:
            mu_obj_eff = 0.0  # no evidence at all -- treat as maximal uncertainty
            chi = 1.0         # force clarification
        else:
            mu_obj_eff = mu_obj
            chi = (mu_det - mu_obj_eff) ** 2

        triggered = unknown or (chi > self.alpha)

        record = {
            'noun': noun, 'mu_det': mu_det, 'mu_obj': mu_obj_eff,
            'unknown_noun': unknown, 'chi': chi, 'alpha_before': self.alpha,
            'triggered': triggered, 'matched_mode': mode_id,
        }

        if triggered:
            F = feedback_fn(noun, fitted_params)
            confirm_entry = self.registry.confirm(fitted_params, noun, F)
            self._update_alpha(F, chi)
            record.update({'F': F, 'registry_action': confirm_entry['action'],
                            'alpha_after': self.alpha})
        else:
            record.update({'F': None, 'registry_action': 'none_autonomous',
                            'alpha_after': self.alpha})

        self.history.append(record)
        return record

    def summary(self):
        n = len(self.history)
        n_trig = sum(1 for r in self.history if r['triggered'])
        return {
            'n_trials': n, 'n_triggered': n_trig,
            'trigger_rate': n_trig / n if n else 0.0,
            'final_alpha': self.alpha,
        }
