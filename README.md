# RankLab — Interactive PageRank & Convergence Analyzer

Python + Streamlit + NumPy + Pandas + Plotly (+ NetworkX for layout only). PageRank is implemented explicitly in `pagerank.py`; no library PageRank is used.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q          # unit tests for the engine
streamlit run app.py
```

https://web-page-ranking.streamlit.app/


## Mathematics (column-stochastic convention)
- `A[i,j] = 1` if page *j* links to page *i* (row = target, column = source). Duplicate links are ignored; self-links allowed.
- `P[:,j] = A[:,j] / outdegree(j)`. A dangling column (no outlinks) is replaced by `1/N` in every entry.
- Iteration: `r(k+1) = d·P·r(k) + (1-d)/N·1`, with `r(0) = 1/N`.
- Stop when `‖r(k+1) − r(k)‖₁ < ε`; if `max_iter` (default 10000) is hit first, the run is reported as **not converged**.
- Every iteration checks that the sum is ≈ 1 and that scores are finite and non-negative. Nothing is silently renormalized.
- Final verification: column sums of P, score sum, full residual `‖dPr + (1-d)/N·1 − r‖₁`, and the raw `‖Pr − r‖₁` (labelled info only; not zero when d < 1).
- For 0 ≤ d < 1 the Google matrix `dP + (1-d)/N·11ᵀ` is strictly positive, so the stationary vector is unique.
- Tolerance experiment reference: direct solve of `(I − dP)r = (1-d)/N·1` (`numpy.linalg.solve`).

## Files
`pagerank.py` engine · `experiments.py` runners · `visualizations.py` Plotly figures · `app.py` UI · `tests/test_pagerank.py`.

## Viva demo guide
1. **Network effect** – Rankings tab → *Save baseline*; add/remove a link in Network Studio; Run; open Experiments → 1.
2. **Damping** – Experiments → 2; note the ranking order and iteration counts differ by network, with no universal trend.
3. **Tolerance** – Experiments → 3; iterations rise as ε shrinks; error vs the direct-solve reference shrinks.
4. **Dangling** – Load *Demo site*; Experiments → 4 shows Contact's corrected column and the effect of adding an outlink.
5. **Verification** – Experiments → 5 or Mathematics Lab step 8.

## Limitations
Nodes are positioned automatically (spring layout), not draggable; editing is via forms. Mathematics Lab shows the first 12 iteration vectors and 30 differences.
