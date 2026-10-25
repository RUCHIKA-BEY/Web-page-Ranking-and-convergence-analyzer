import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pytest
from pagerank import *

def build(pages, links, **kw):
    return run_pagerank(adjacency_matrix(pages, links), pages, **kw)

def test_adjacency_orientation():
    A = adjacency_matrix(["a", "b"], [("a", "b")])
    assert A[1, 0] == 1 and A[0, 1] == 0           # a -> b stored at row b, column a

def test_duplicate_links_ignored():
    assert adjacency_matrix(["a", "b"], [("a", "b")] * 3).sum() == 1

def test_columns_stochastic_and_dangling_uniform():
    A = adjacency_matrix(["a", "b", "c"], [("a", "b"), ("a", "c")])
    P, dangling = transition_matrix(A)
    assert dangling == [1, 2]
    assert np.allclose(P.sum(axis=0), 1)
    assert np.allclose(P[:, 1], 1 / 3)
    assert np.allclose(P[:, 0], [0, .5, .5])

def test_three_page_matches_direct_solve_and_stationary_eq():
    pages = ["A", "B", "C"]
    res = build(pages, [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")], tol=1e-12)
    assert res.converged
    assert np.allclose(res.r, reference_solution(res.P, 0.85), atol=1e-10)
    assert abs(res.r.sum() - 1) < 1e-12
    checks = {c["check"]: c for c in verify(res)}
    assert checks["Full damped residual small"]["passed"]
    assert checks["Raw ||Pr - r||_1 (info only)"]["value"] > 1e-3   # not zero when d<1

def test_single_dangling_page():
    res = build(["solo"], [])
    assert res.converged and np.allclose(res.r, [1.0])

def test_disconnected_components():
    res = build(["a", "b", "c", "d"], [("a", "b"), ("b", "a"), ("c", "d")])
    assert res.converged and abs(res.r.sum() - 1) < 1e-12 and (res.r > 0).all()

def test_self_link_and_cycle():
    res = build(["a", "b"], [("a", "a"), ("a", "b"), ("b", "a")])
    assert res.converged and abs(res.r.sum() - 1) < 1e-12

def test_max_iterations_reports_not_converged():
    res = build(["a", "b", "c"], [("a", "b"), ("a", "c"), ("b", "c"), ("c", "a")], d=0.95, tol=1e-15, max_iter=2)
    assert not res.converged and res.iterations == 2

def test_looser_tolerance_never_needs_more_iterations():
    pages, links = ["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")]
    its = [build(pages, links, tol=t).iterations for t in (1e-3, 1e-6, 1e-8)]
    assert its == sorted(its)

@pytest.mark.parametrize("d", [-0.1, 1.0, 1.5])
def test_invalid_damping(d):
    with pytest.raises(ValueError):
        build(["a"], [], d=d)

def test_empty_and_unknown_page_errors():
    with pytest.raises(ValueError): adjacency_matrix([], [])
    with pytest.raises(ValueError): adjacency_matrix(["a"], [("a", "z")])

def test_scores_normalised_every_iteration():
    res = build(["a", "b", "c"], [("a", "b")])
    assert all(abs(h.sum() - 1) < 1e-12 for h in res.history)
