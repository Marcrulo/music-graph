"""Helpers shared by the extractors: summary statistics and named flattening."""

import re

import numpy as np


def stats(M, prefix="") -> dict:
    """mean/std/p10/p90 per row."""
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    if M.shape[1] == 0:
        M = np.zeros((M.shape[0], 1))
    return {f"{prefix}mean": M.mean(1), f"{prefix}std": M.std(1),
            f"{prefix}p10": np.percentile(M, 10, axis=1), f"{prefix}p90": np.percentile(M, 90, axis=1)}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def flatten(value, prefix: str, names: list, values: list) -> None:
    """Nested dicts / arrays -> parallel lists of dimension names and floats."""
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(v, f"{prefix}.{k}", names, values)
        return
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        names.append(prefix)
        values.append(float(arr))
    else:
        for i, x in enumerate(arr.ravel()):
            names.append(f"{prefix}[{i}]")
            values.append(float(x))


def run_features(groups: dict[str, dict], extractor: str) -> tuple[list[str], np.ndarray]:
    """groups: {group: {feature name: callable}} -> (names, float32 vector).

    Names are '<group>/<feature>.<key>[i]'; the group prefix lets embeddings weight
    categories separately. Any failing feature fails the whole track (dims must be fixed).
    """
    names, values = [], []
    for group, feats in groups.items():
        for fname, fn in feats.items():
            try:
                out = fn()
            except Exception as e:
                raise RuntimeError(f"{extractor}: {group}/{fname}: {type(e).__name__}: {e}") from e
            flatten(out, f"{group}/{slug(fname)}", names, values)
    return names, np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
