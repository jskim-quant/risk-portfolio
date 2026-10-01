"""Standard Formula premium and reserve risk component (illustrative case)."""
from dataclasses import replace

import numpy as np
import pytest

from solvency_model import SegmentInputs, premium_reserve_scr


def test_single_segment_and_empty_volume():
    row = SegmentInputs(1, 100, 120, 5, 2, 10, 60, 10)
    result = premium_reserve_scr([row])
    assert result['segments'][0]['premium_volume'] == 130 and result['segments'][0]['reserve_volume'] == 50
    assert result['scr'] == pytest.approx(3 * np.sqrt((.08 * 130) ** 2 + (.08 * 130) * (.09 * 50) + (.09 * 50) ** 2))
    assert premium_reserve_scr([SegmentInputs(1, 0, 0, 0, 0, 0, -5, 0)])['scr'] == 0
    with pytest.raises(ValueError):
        premium_reserve_scr([replace(row, segment=99)])
    with pytest.raises(ValueError):
        premium_reserve_scr([row, row])


def test_correlation_scaling_and_order_invariance():
    a, b = SegmentInputs(1, 100, 110, 8, 2, 10, 90, 15), SegmentInputs(5, 60, 55, 5, 1, 20, 130, 20)
    out = premium_reserve_scr([a, b])
    stand = [x['standalone_scr'] for x in out['segments']]
    assert out['scr'] == pytest.approx(np.sqrt(stand[0] ** 2 + stand[0] * stand[1] + stand[1] ** 2))
    assert out['scr'] == pytest.approx(premium_reserve_scr([b, a])['scr'])
    scaled = [SegmentInputs(x.segment, *(2 * v for k, v in vars(x).items() if k != 'segment')) for x in (a, b)]
    assert premium_reserve_scr(scaled)['scr'] == pytest.approx(2 * out['scr'])
    assert out['scr'] <= sum(stand) + 1e-9          # correlation 0.5 < 1, so aggregation never exceeds the sum
