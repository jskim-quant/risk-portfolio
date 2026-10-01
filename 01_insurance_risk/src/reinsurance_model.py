"""Reinsurance transformations. Every function returns (retained, ceded) with retained + ceded = gross exactly."""
from __future__ import annotations

import numpy as np


def quota_share(losses: np.ndarray, share: float) -> tuple[np.ndarray, np.ndarray]:
    """Cede a fixed fraction of every loss."""
    if not 0 <= share <= 1:
        raise ValueError("share must be in [0, 1]")
    losses = np.asarray(losses, dtype=float)
    ceded = share * losses
    return losses - ceded, ceded


def per_occurrence_xol(claims: np.ndarray, retention: float, limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Excess of loss on each claim: the reinsurer pays min(max(claim - retention, 0), limit)."""
    if retention < 0:
        raise ValueError("retention must be nonnegative")
    claims = np.asarray(claims, dtype=float)
    ceded = np.clip(claims - retention, 0.0, None if limit is None else limit)
    return claims - ceded, ceded


def aggregate_xol(annual_losses: np.ndarray, attachment: float, limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stop-loss on an annual total: layer `limit` xs `attachment` (unlimited when limit is None)."""
    return per_occurrence_xol(annual_losses, attachment, limit)
