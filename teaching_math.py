"""Small, UI-independent calculation helpers for the Streamlit teaching extension.

The functions intentionally preserve the original model formulation.  They only
make deterministic expected flows and IPF iteration state explicit and testable.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class IPFResult:
    matrix: np.ndarray
    history: List[dict]
    converged: bool
    status: str
    iterations: int
    error: float
    message: str


def expected_origin_flows(O, attractiveness, dist, beta):
    """Return expected OD flows, probabilities and unnormalised weights.

    This is the deterministic counterpart of the original row-wise calculation.
    The removed ``mode_job_pref`` factor was a mode-wide scalar and therefore
    cancelled exactly during row normalisation.
    """
    O = np.asarray(O, dtype=float)
    attractiveness = np.asarray(attractiveness, dtype=float)
    dist = np.asarray(dist, dtype=float)
    if O.ndim != 1 or attractiveness.ndim != 1:
        raise ValueError("O and attractiveness must be one-dimensional")
    if dist.shape != (len(O), len(attractiveness)):
        raise ValueError("Distance matrix dimensions do not match O and attractiveness")
    if np.any(~np.isfinite(O)) or np.any(O < 0):
        raise ValueError("Origin totals must be finite and non-negative")
    if np.any(~np.isfinite(attractiveness)) or np.any(attractiveness < 0):
        raise ValueError("Attractiveness must be finite and non-negative")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("Beta must be finite and non-negative")

    weights = attractiveness[np.newaxis, :] * np.exp(-beta * dist)
    denominators = weights.sum(axis=1)
    invalid = (O > 0) & (denominators <= 0)
    if np.any(invalid):
        rows = np.where(invalid)[0].tolist()
        raise ValueError(f"Zero probability denominator for origin rows {rows}")
    probabilities = np.divide(
        weights,
        denominators[:, np.newaxis],
        out=np.zeros_like(weights),
        where=denominators[:, np.newaxis] > 0,
    )
    expected = O[:, np.newaxis] * probabilities
    return expected, probabilities, weights


def poisson_sample(expected, seed=2025):
    """Sample integer flows with an explicit, reproducible seed."""
    expected = np.asarray(expected, dtype=float)
    if np.any(~np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("Expected flows must be finite and non-negative")
    rng = np.random.default_rng(int(seed))
    return rng.poisson(expected).astype(float)


def scale_destination_targets(O, D):
    """Explicitly scale D so its total equals the total of O."""
    O = np.asarray(O, dtype=float)
    D = np.asarray(D, dtype=float)
    origin_total = O.sum()
    destination_total = D.sum()
    if destination_total <= 0:
        if origin_total <= 0:
            return D.copy(), 1.0
        raise ValueError("Cannot scale destination targets whose total is zero")
    factor = origin_total / destination_total
    return D * factor, factor


def teaching_destination_targets(O, jobs):
    """Create an independent classroom destination target from job shares.

    This is an explicit teaching scenario, not an empirical calibration of the
    original LUTI model: D_j = sum(O) * Jobs_j / sum(Jobs).
    """
    O = np.asarray(O, dtype=float)
    jobs = np.asarray(jobs, dtype=float)
    if O.ndim != 1 or jobs.ndim != 1 or len(O) != len(jobs):
        raise ValueError("O and jobs must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(O)) or np.any(O < 0):
        raise ValueError("Origin totals must be finite and non-negative")
    if np.any(~np.isfinite(jobs)) or np.any(jobs < 0):
        raise ValueError("Jobs must be finite and non-negative")
    if jobs.sum() <= 0:
        if O.sum() <= 0:
            return np.zeros_like(jobs)
        raise ValueError("Teaching destination targets require a positive jobs total")
    return O.sum() * jobs / jobs.sum()


def selected_od_snapshot(matrix, origin, destination):
    """Return the selected OD value and matching row/column totals."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("OD matrix must be two-dimensional")
    origin = int(origin)
    destination = int(destination)
    if not (0 <= origin < matrix.shape[0] and 0 <= destination < matrix.shape[1]):
        raise IndexError("Selected OD is outside the matrix")
    return {
        "origin": origin,
        "destination": destination,
        "flow": float(matrix[origin, destination]),
        "row_sum": float(matrix[origin, :].sum()),
        "column_sum": float(matrix[:, destination].sum()),
    }


def flow_highlight_payload(matrix, coords, origin, destination):
    """Build the map highlight from the same selected OD snapshot as formulae."""
    coords = np.asarray(coords, dtype=float)
    snapshot = selected_od_snapshot(matrix, origin, destination)
    if coords.ndim != 2 or coords.shape[0] != np.asarray(matrix).shape[0] or coords.shape[1] < 2:
        raise ValueError("Coordinates must contain at least x and y for every zone")
    i, j = snapshot['origin'], snapshot['destination']
    return {
        **snapshot,
        "x": [float(coords[i, 0]), float(coords[j, 0])],
        "y": [float(coords[i, 1]), float(coords[j, 1])],
    }


def select_display_matrix(origin_matrix, ipf_result=None, frame_index=None):
    """Select one authoritative matrix for formulae and maps.

    Failed IPF runs deliberately fall back to the valid origin-constrained
    matrix instead of exposing a fabricated all-zero result.
    """
    origin_matrix = np.asarray(origin_matrix, dtype=float)
    if ipf_result is None:
        return origin_matrix.copy(), "Origin-Constrained", True
    if not ipf_result.converged:
        return origin_matrix.copy(), "Origin-Constrained fallback (IPF failed)", False
    if frame_index is None:
        return ipf_result.matrix.copy(), "Final converged OD matrix (trips)", True
    frame_index = int(np.clip(frame_index, 0, len(ipf_result.history) - 1))
    frame = ipf_result.history[frame_index]
    if frame["phase"] == "initial":
        label = "Initial weights / 初始权重（无出行次数单位）"
    else:
        label = (
            f"Current IPF frame: iteration {frame['iteration']} "
            f"{frame['phase']} adjustment (trips)"
        )
    return frame["matrix"].copy(), label, True


def prune_zone_edits(edits, zone_count):
    """Return valid, normalised per-zone edits for the current city size."""
    cleaned = {}
    for zone, values in dict(edits or {}).items():
        zone = int(zone)
        if 0 <= zone < int(zone_count):
            population = int(values['pop'])
            jobs = int(values['jobs'])
            if population < 0 or jobs < 0:
                raise ValueError("Population and jobs edits must be non-negative")
            cleaned[zone] = {'pop': population, 'jobs': jobs}
    return cleaned


def update_zone_edit(edits, zone, population, jobs, zone_count):
    """Persist one zone edit without discarding edits for other zones."""
    cleaned = prune_zone_edits(edits, zone_count)
    zone = int(zone)
    if not 0 <= zone < int(zone_count):
        raise IndexError("Edited zone is outside the current city")
    if int(population) < 0 or int(jobs) < 0:
        raise ValueError("Population and jobs edits must be non-negative")
    cleaned[zone] = {'pop': int(population), 'jobs': int(jobs)}
    return cleaned


def apply_zone_edits(population, jobs, edits):
    """Apply saved edits to copies of the base population and jobs arrays."""
    population = np.asarray(population).copy()
    jobs = np.asarray(jobs).copy()
    cleaned = prune_zone_edits(edits, len(population))
    if len(population) != len(jobs):
        raise ValueError("Population and jobs arrays must have equal length")
    for zone, values in cleaned.items():
        population[zone] = values['pop']
        jobs[zone] = values['jobs']
    return population, jobs


def fast_playback_frames(history, key_iterations=(1, 2, 5, 10, 20, 50)):
    """Return real history indices for a concise classroom playback.

    The first row-adjustment frame is retained to show the transition from
    unitless weights to trips. Later key nodes use completed column-adjustment
    frames, followed by the actual final frame.
    """
    if not history:
        return []
    selected = [0]
    for index, frame in enumerate(history):
        if frame['phase'] == 'row' and frame['iteration'] == 1:
            selected.append(index)
            break
    wanted = set(int(value) for value in key_iterations)
    for index, frame in enumerate(history):
        if frame['phase'] == 'column' and int(frame['iteration']) in wanted:
            selected.append(index)
    selected.append(len(history) - 1)
    return sorted(set(selected))


def playback_neighbor(current_frame, available_frames, direction):
    """Move to the previous or next real frame in a playback sequence."""
    available = sorted(set(int(value) for value in available_frames))
    if not available:
        raise ValueError("Playback sequence cannot be empty")
    current_frame = int(current_frame)
    if direction > 0:
        later = [value for value in available if value > current_frame]
        return later[0] if later else available[-1]
    earlier = [value for value in available if value < current_frame]
    return earlier[-1] if earlier else available[0]


def playback_frame_snapshot(history, frame_index):
    """Expose matrix, error, phase, iteration and unit from one true frame."""
    frame_index = int(frame_index)
    if not 0 <= frame_index < len(history):
        raise IndexError("Playback frame is outside the history")
    frame = history[frame_index]
    return {
        "frame_index": frame_index,
        "matrix": frame['matrix'].copy(),
        "error": float(frame['error']),
        "row_error": float(frame['row_error']),
        "col_error": float(frame['col_error']),
        "phase": frame['phase'],
        "iteration": int(frame['iteration']),
        "unit": "unitless weight" if frame['phase'] == 'initial' else "trips",
    }


def _errors(matrix, O, D):
    row_error = float(np.max(np.abs(matrix.sum(axis=1) - O))) if len(O) else 0.0
    col_error = float(np.max(np.abs(matrix.sum(axis=0) - D))) if len(D) else 0.0
    return row_error, col_error, max(row_error, col_error)


def _history_item(matrix, phase, iteration, O, D):
    row_error, col_error, error = _errors(matrix, O, D)
    return {
        "matrix": matrix.copy(),
        "phase": phase,
        "iteration": iteration,
        "row_error": row_error,
        "col_error": col_error,
        "error": error,
    }


def run_ipf(O, D, F_base, max_iter=500, tol=1e-6):
    """Run IPF/Furness and retain initial, row and column adjustment frames."""
    O = np.asarray(O, dtype=float)
    D = np.asarray(D, dtype=float)
    F = np.asarray(F_base, dtype=float).copy()
    if F.shape != (len(O), len(D)):
        raise ValueError("Base matrix dimensions must match O and D")
    if np.any(~np.isfinite(F)) or np.any(F < 0):
        raise ValueError("Base matrix must be finite and non-negative")
    if np.any(~np.isfinite(O)) or np.any(O < 0) or np.any(~np.isfinite(D)) or np.any(D < 0):
        raise ValueError("Targets must be finite and non-negative")

    history = [_history_item(F, "initial", 0, O, D)]
    total_gap = float(abs(O.sum() - D.sum()))
    if total_gap > tol:
        return IPFResult(
            F, history, False, "total_mismatch", 0, total_gap,
            f"Origin total {O.sum():.6f} differs from destination total {D.sum():.6f}.",
        )
    impossible_rows = np.where((O > tol) & (F.sum(axis=1) <= 0))[0]
    impossible_cols = np.where((D > tol) & (F.sum(axis=0) <= 0))[0]
    if len(impossible_rows) or len(impossible_cols):
        return IPFResult(
            F, history, False, "infeasible", 0, float("inf"),
            f"Structural zeros make rows {impossible_rows.tolist()} or columns {impossible_cols.tolist()} infeasible.",
        )

    for iteration in range(1, int(max_iter) + 1):
        row_sums = F.sum(axis=1)
        row_factors = np.divide(O, row_sums, out=np.zeros_like(O), where=row_sums > 0)
        F = F * row_factors[:, np.newaxis]
        history.append(_history_item(F, "row", iteration, O, D))

        col_sums = F.sum(axis=0)
        col_factors = np.divide(D, col_sums, out=np.zeros_like(D), where=col_sums > 0)
        F = F * col_factors[np.newaxis, :]
        frame = _history_item(F, "column", iteration, O, D)
        history.append(frame)
        if frame["error"] <= tol:
            return IPFResult(
                F, history, True, "converged", iteration, frame["error"],
                f"Converged after {iteration} full iterations.",
            )

    error = history[-1]["error"]
    return IPFResult(
        F, history, False, "max_iterations", int(max_iter), error,
        f"Stopped at {max_iter} iterations without convergence; maximum error is {error:.6g}.",
    )
