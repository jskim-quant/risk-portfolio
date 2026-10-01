"""Standard Formula premium/reserve component for Annex II segments 1 and 5.

Basis: Regulation (EU) 2015/35, Articles 115–117 and Annexes II/IV,
Annex II as replaced by 2019/981. Applicable date: 2026-09-30.
Only default geographic factor DIV=1 and Article 116(3) volumes are supported.
All monetary inputs must be in the same currency and unit at one valuation date.
"""
from dataclasses import dataclass, asdict
import numpy as np

BASIS_DATE = '2026-09-30'
RESULT_NAME = 'Illustrative premium and reserve underwriting-risk SCR component (Standard Formula)'
PARAMETERS = {1: (0.10, 0.09, 0.80), 5: (0.14, 0.11, 0.80)}
SEGMENT_NAMES = {1: 'Motor vehicle liability', 5: 'General liability'}
CORRELATION = np.array([[1.0, 0.5], [0.5, 1.0]])  # Annex IV, rows/columns 1 and 5

@dataclass
class SegmentInputs:
    segment: int
    next_12m_net_earned: float
    last_12m_net_earned: float
    future_existing_pv: float
    future_new_short_pv: float
    future_new_long_pv: float
    gross_claims_best_estimate: float
    eligible_recoverables: float


def premium_reserve_scr(inputs: list[SegmentInputs]) -> dict:
    if not inputs or len({x.segment for x in inputs}) != len(inputs):
        raise ValueError('Provide one row per supported segment, without duplicates')
    rows = []
    for item in inputs:
        if item.segment not in PARAMETERS:
            raise ValueError('Only Annex II segments 1 and 5 are implemented')
        values = [v for k, v in asdict(item).items() if k != 'segment']
        if not np.isfinite(values).all() or min(values[:5]) < 0 or item.eligible_recoverables < 0:
            raise ValueError('Premium inputs and recoverables must be finite and nonnegative')
        vp = max(item.next_12m_net_earned, item.last_12m_net_earned) + item.future_existing_pv
        vp += item.future_new_short_pv + 0.30 * item.future_new_long_pv
        vr = max(item.gross_claims_best_estimate - item.eligible_recoverables, 0.0)
        gross_sigma, reserve_sigma, adjustment = PARAMETERS[item.segment]
        a, b = gross_sigma * adjustment * vp, reserve_sigma * vr
        weighted_sigma = float(np.sqrt(a*a + a*b + b*b))
        volume = vp + vr  # DIV=1: no geographic diversification credit
        rows.append(dict(segment=item.segment, premium_volume=vp, reserve_volume=vr,
            volume=volume, premium_sigma=gross_sigma * adjustment, reserve_sigma=reserve_sigma,
            sigma=weighted_sigma / volume if volume else 0.0,
            weighted_sigma=weighted_sigma, standalone_scr=3 * weighted_sigma))
    indices = [[1, 5].index(r['segment']) for r in rows]
    corr = CORRELATION[np.ix_(indices, indices)]
    weighted = np.array([r['weighted_sigma'] for r in rows])
    volume = sum(r['volume'] for r in rows)
    charge = float(3 * np.sqrt(weighted @ corr @ weighted))
    return dict(name=RESULT_NAME, basis_date=BASIS_DATE, currency='EUR', unit='million',
                volume=volume, sigma=charge / (3 * volume) if volume else 0.0,
                scr=charge, segments=rows, correlation=corr.tolist())
