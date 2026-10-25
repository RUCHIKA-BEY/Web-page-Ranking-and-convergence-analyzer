"""Plotly figures: network, convergence, ranking and comparison charts."""
from __future__ import annotations
import math
import networkx as nx
import numpy as np
import plotly.graph_objects as go

NAVY, PANEL, TEXT, ACCENT, VIOLET, WARN = "#0b1220", "#131c31", "#e6eaf5", "#5b8cff", "#8b5cf6", "#f59e0b"


def _style(fig, height=420, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor=NAVY, plot_bgcolor=PANEL,
                      font=dict(color=TEXT), height=height, margin=dict(l=40, r=20, t=50 if title else 20, b=40),
                      title=title)
    return fig


def layout_positions(pages, links, seed=7):
    G = nx.DiGraph(); G.add_nodes_from(pages); G.add_edges_from(links)
    return nx.spring_layout(G, seed=seed, k=1.4 / max(1, math.sqrt(len(pages)))) if len(pages) > 1 else {pages[0]: (0, 0)}


def network_figure(pages, links, pos, scores=None, dangling=(), selected=None):
    fig = go.Figure()
    for s, t in links:                                    # arrows drawn as annotations
        (x0, y0), (x1, y1) = pos[s], pos[t]
        if s == t:
            fig.add_annotation(x=x0 + .06, y=y0 + .09, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowcolor="#7a86a8")
            continue
        sh = 0.07                                         # shorten so the arrowhead clears the node
        dx, dy = x1 - x0, y1 - y0; L = math.hypot(dx, dy) or 1
        fig.add_annotation(x=x1 - dx / L * sh, y=y1 - dy / L * sh, ax=x0 + dx / L * sh, ay=y0 + dy / L * sh,
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.3, arrowwidth=1.4, arrowcolor="#7a86a8")
    n = len(pages)
    vals = np.array(scores) if scores is not None else np.full(n, 1 / n)
    size = 22 + 70 * (vals / vals.max() if vals.max() > 0 else vals)
    out = {p: sum(1 for s, _ in links if s == p) for p in pages}
    inn = {p: sum(1 for _, t in links if t == p) for p in pages}
    hover = [f"<b>{p}</b><br>PageRank: {v:.4f}<br>in: {inn[p]} · out: {out[p]}"
             f"{'<br>dangling page' if out[p] == 0 else ''}" for p, v in zip(pages, vals)]
    line_c = [WARN if i in dangling else "#0b1220" for i in range(n)]
    line_c = ["#ffffff" if p == selected else c for p, c in zip(pages, line_c)]
    fig.add_trace(go.Scatter(x=[pos[p][0] for p in pages], y=[pos[p][1] for p in pages], mode="markers+text",
                             text=pages, textposition="bottom center", hovertext=hover, hoverinfo="text",
                             marker=dict(size=size, color=vals, colorscale=[[0, "#2b3a67"], [1, VIOLET]],
                                         line=dict(width=3, color=line_c), colorbar=dict(title="score") if scores is not None else None)))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False, scaleanchor="x")
    return _style(fig, 520).update_layout(showlegend=False, plot_bgcolor=NAVY)


def convergence_figure(res, stop_marker=True):
    it = list(range(1, len(res.deltas) + 1))
    fig = go.Figure(go.Scatter(x=it, y=res.deltas, mode="lines+markers", name="L1 difference",
                               line=dict(color=ACCENT), hovertemplate="iter %{x}<br>Δ=%{y:.3e}<extra></extra>"))
    fig.add_hline(y=res.tol, line_dash="dash", line_color=WARN, annotation_text=f"tolerance {res.tol:g}")
    if stop_marker and res.converged:
        fig.add_vline(x=res.iterations, line_dash="dot", line_color=VIOLET, annotation_text=f"stopped at {res.iterations}")
    fig.update_yaxes(type="log", title="‖r⁽ᵏ⁺¹⁾ − r⁽ᵏ⁾‖₁"); fig.update_xaxes(title="iteration")
    return _style(fig, 400, "Convergence")


def ranking_bar(pages, scores, title="PageRank scores"):
    order = np.argsort(scores)
    fig = go.Figure(go.Bar(x=[scores[i] for i in order], y=[pages[i] for i in order], orientation="h",
                           marker_color=ACCENT, hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    return _style(fig, 60 + 40 * len(pages), title)


def grouped_scores(pages, series: dict[str, np.ndarray], title):
    fig = go.Figure([go.Bar(name=k, x=pages, y=v) for k, v in series.items()])
    fig.update_layout(barmode="group"); fig.update_yaxes(title="score")
    return _style(fig, 420, title)


def heatmap(pages, labels, matrix, title):
    fig = go.Figure(go.Heatmap(z=np.array(matrix).T, x=labels, y=pages, colorscale="Purples",
                               hovertemplate="%{y} @ %{x}: %{z:.4f}<extra></extra>"))
    return _style(fig, 60 + 45 * len(pages), title)
