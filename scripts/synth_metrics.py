import numpy as np
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def pinball_metrics(q, y, scale):
    q = np.asarray(q, float); y = np.asarray(y, float); scale = np.asarray(scale, float)
    ok = np.isfinite(q).all(1) & np.isfinite(y)
    q, y, scale = q[ok], y[ok], scale[ok]
    diff = y[:, None] - q
    pl = 2 * np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(1)
    return dict(MASE=float(np.mean(np.abs(q[:, 4] - y) / scale)), CRPS_s=float(np.mean(pl / scale)),
                WQL=float(pl.sum() / (np.abs(y).sum() + 1e-12)), COV80=float(np.mean((y >= q[:, 0]) & (y <= q[:, 8]))), n=int(ok.sum()))
