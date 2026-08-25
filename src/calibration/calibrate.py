from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def fit_calibrator(
    base_estimator: Any,
    X_val: Any,
    y_val: np.ndarray,
    *,
    method: str = "isotonic",
) -> Any:
    """
    Fit probability calibration on validation data only.

    Uses a frozen/prefit estimator so the held-out test set remains untouched
    and training weights are not refit during calibration.
    """
    y_val = np.asarray(y_val)

    # scikit-learn >=1.6: FrozenEstimator replaces cv='prefit'
    try:
        from sklearn.frozen import FrozenEstimator

        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_estimator),
            method=method,
            cv=None,
        )
        calibrator.fit(X_val, y_val)
        return calibrator
    except Exception:
        pass

    # Older API
    try:
        calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=method,
            cv="prefit",
        )
        calibrator.fit(X_val, y_val)
        return calibrator
    except Exception as exc:
        raise RuntimeError(f"Calibration unavailable with current scikit-learn: {exc}") from exc
