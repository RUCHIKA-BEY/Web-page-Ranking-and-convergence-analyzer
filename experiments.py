"""Experiment runners. Every result comes from real runs of pagerank.run_pagerank."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pagerank import adjacency_matrix, run_pagerank, reference_solution

DAMPING_VALUES = (0.5, 0.7, 0.85, 0.95)
TOLERANCES = (1e-3, 1e-6, 1e-8)


def run(pages, links, d=0.85, tol=1e-6, max_iter=10000):
    return run_pagerank(adjacency_matrix(pages, links), pages, d, tol, max_iter)


def ranks(scores):
    order = np.argsort(-np.asarray(scores), kind="stable")
    out = np.empty(len(scores), dtype=int); out[order] = np.arange(1, len(scores) + 1)
    return out


def compare_runs(pages, before, after):
    """Experiment 1 / 4: per-page score and rank change between two runs on the same page set."""
    rb, ra = ranks(before.r), ranks(after.r)
    return pd.DataFrame({"Page": pages, "Old score": before.r, "New score": after.r,
                         "Δ score": after.r - before.r, "Old rank": rb, "New rank": ra,
                         "Rank change (+ = up)": rb - ra}).sort_values("New rank").reset_index(drop=True)


def damping_experiment(pages, links, tol=1e-6):
    runs = {d: run(pages, links, d, tol) for d in DAMPING_VALUES}
    summary = pd.DataFrame([{"d": d, "Iterations": r.iterations, "Final L1 diff": r.final_delta,
                             "Converged": r.converged, "Top page": pages[int(np.argmax(r.r))],
                             "Ranking": " > ".join(pages[i] for i in np.argsort(-r.r, kind="stable"))}
                            for d, r in runs.items()])
    scores = pd.DataFrame({f"d={d}": r.r for d, r in runs.items()}, index=pages)
    return runs, summary, scores


def tolerance_experiment(pages, links, d=0.85):
    runs = {t: run(pages, links, d, t) for t in TOLERANCES}
    P = next(iter(runs.values())).P
    ref = reference_solution(P, d)       # independent method: direct solve of (I - dP) r = (1-d)/N 1
    summary = pd.DataFrame([{"Tolerance": t, "Iterations": r.iterations, "Final L1 diff": r.final_delta,
                             "Converged": r.converged, "L1 error vs reference": float(np.abs(r.r - ref).sum())}
                            for t, r in runs.items()])
    return runs, summary, ref
