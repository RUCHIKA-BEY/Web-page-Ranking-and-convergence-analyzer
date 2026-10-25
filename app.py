"""RankLab - interactive PageRank laboratory (Streamlit interface)."""
import numpy as np
import pandas as pd
import streamlit as st
from pagerank import adjacency_matrix, google_matrix, transition_matrix, verify
import experiments as ex
import visualizations as viz

st.set_page_config(page_title="RankLab", page_icon="🔗", layout="wide")
st.markdown("""<style>
.stApp{background:#0b1220;color:#e6eaf5}
[data-testid=stSidebar]{background:#0f1830}
[data-testid=stMetric]{background:#131c31;border:1px solid #24304f;border-radius:12px;padding:12px 16px}
h1,h2,h3{letter-spacing:-.01em}
</style>""", unsafe_allow_html=True)

PRESETS = {
    "Demo site (6 pages)": (["Home", "Research", "Tutorials", "Projects", "Resources", "Contact"],
        [("Home", "Research"), ("Home", "Tutorials"), ("Home", "Projects"), ("Home", "Contact"),
         ("Research", "Tutorials"), ("Research", "Resources"), ("Tutorials", "Home"), ("Tutorials", "Projects"),
         ("Projects", "Research"), ("Projects", "Resources"), ("Resources", "Research")]),
    "Three-page hand example": (["A", "B", "C"], [("A", "B"), ("A", "C"), ("B", "C"), ("C", "A")]),
    "Disconnected components": (["A", "B", "C", "D", "E"], [("A", "B"), ("B", "A"), ("C", "D"), ("D", "E"), ("E", "C")]),
    "Self-link and dangling": (["Hub", "Spam", "Leaf", "Sink"], [("Hub", "Leaf"), ("Hub", "Sink"), ("Spam", "Spam"), ("Spam", "Hub"), ("Leaf", "Hub")]),
}


def load_preset(name):
    pages, links = PRESETS[name]
    st.session_state.update(pages=list(pages), links=list(links), result=None, sig=None, base=None, step=0)


if "pages" not in st.session_state:
    load_preset("Demo site (6 pages)")
    st.session_state.update(d=0.85, tol=1e-6, max_iter=10000, selected=None)

S = st.session_state

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🔗 RankLab")
    st.caption("Interactive PageRank & convergence analyzer")
    preset = st.selectbox("Preset network", list(PRESETS))
    if st.button("Load preset / reset", width="stretch"):
        load_preset(preset); st.rerun()
    S.d = st.slider("Damping factor d", 0.0, 0.99, S.d, 0.01,
                    help="Probability the surfer follows a link; 1-d is the chance of jumping to a random page.")
    S.tol = st.select_slider("Tolerance ε", [1e-3, 1e-6, 1e-8], S.tol if S.tol in (1e-3, 1e-6, 1e-8) else 1e-6,
                             format_func=lambda v: f"{v:.0e}", help="Stop when the L1 change between iterations drops below ε.")
    S.max_iter = st.number_input("Max iterations", 1, 100000, S.max_iter)
    run_clicked = st.button("▶ Run PageRank", type="primary", width="stretch")

pages, links = S.pages, S.links


def signature():
    return (tuple(pages), tuple(sorted(links)), S.d, S.tol, S.max_iter)


if run_clicked:
    try:
        S.result = ex.run(pages, links, S.d, S.tol, int(S.max_iter)); S.sig = signature(); S.step = 0
    except ValueError as e:
        st.sidebar.error(str(e))

res = S.result
fresh = res is not None and S.sig == signature()
if res is not None and not fresh:
    st.warning("⚠ The graph or settings changed since the last run. Results below are from the previous configuration — press **Run PageRank**.")

tabs = st.tabs(["Overview", "Network Studio", "Rankings", "Mathematics Lab", "Convergence", "Experiments"])
idx = {p: i for i, p in enumerate(pages)}

# ---------------- overview ----------------
with tabs[0]:
    st.header("RankLab")
    st.write("**PageRank** scores a page by how likely a random surfer is to land on it. The surfer follows a random "
             "link with probability *d* and otherwise jumps to a random page. The scores are the long-run visit "
             "frequencies, found by repeating `r ← d·P·r + (1−d)/N` until they stop changing.")
    c = st.columns(5)
    c[0].metric("Pages", len(pages)); c[1].metric("Links", len(links))
    c[2].metric("Damping / ε", f"{S.d:.2f} / {S.tol:.0e}")
    c[3].metric("Top page", pages[int(np.argmax(res.r))] if res else "—")
    c[4].metric("Last run", (f"{res.iterations} it · " + ("converged" if res.converged else "NOT converged")) if res else "not run")
    if res is None:
        st.info("Press **▶ Run PageRank** in the sidebar to compute scores for this network.")

# ---------------- network studio ----------------
with tabs[1]:
    st.subheader("Network Studio")
    left, right = st.columns([3, 2])
    with right:
        S.selected = st.selectbox("Select a page", [None] + pages, format_func=lambda x: "—" if x is None else x)
        if S.selected:
            outs = [t for s, t in links if s == S.selected]; ins = [s for s, t in links if t == S.selected]
            st.markdown(f"**{S.selected}** · out-degree {len(outs)}{' · ⚠ dangling' if not outs else ''}")
            st.write("Links to:", ", ".join(outs) or "—"); st.write("Linked from:", ", ".join(ins) or "—")
            if fresh: st.write(f"PageRank: **{res.r[idx[S.selected]]:.5f}**")
        with st.expander("Add / rename / delete pages", expanded=False):
            with st.form("add_page", clear_on_submit=True):
                name = st.text_input("New page name")
                if st.form_submit_button("Add page") and name.strip():
                    if name.strip() in pages: st.error("That name already exists.")
                    else: pages.append(name.strip()); st.rerun()
            if S.selected:
                with st.form("rename"):
                    new = st.text_input("Rename selected page", S.selected)
                    if st.form_submit_button("Rename") and new.strip() and new != S.selected:
                        if new in pages: st.error("That name already exists.")
                        else:
                            S.pages = [new if p == S.selected else p for p in pages]
                            S.links = [(new if s == S.selected else s, new if t == S.selected else t) for s, t in links]
                            S.selected = new; st.rerun()
                if st.checkbox(f"Confirm: delete “{S.selected}” and its links") and st.button("Delete page"):
                    S.pages = [p for p in pages if p != S.selected]
                    S.links = [(s, t) for s, t in links if S.selected not in (s, t)]
                    S.selected = None; st.rerun()
        with st.expander("Add / remove links"):
            if len(pages) >= 1:
                a, b = st.columns(2)
                src = a.selectbox("From", pages, key="src"); dst = b.selectbox("To", pages, key="dst")
                add, rem = st.columns(2)
                if add.button("Add link"):
                    if (src, dst) in links: st.error("Link already exists.")
                    else: links.append((src, dst)); st.rerun()
                if rem.button("Remove link"):
                    if (src, dst) in links: links.remove((src, dst)); st.rerun()
                    else: st.error("No such link.")
        st.caption("Node size/colour = PageRank · amber ring = dangling page · white ring = selected.")
    with left:
        pos = viz.layout_positions(pages, links)
        dang = res.dangling if fresh else [i for i, p in enumerate(pages) if not any(s == p for s, _ in links)]
        st.plotly_chart(viz.network_figure(pages, links, pos, res.r if fresh else None, dang, S.selected),
                        width="stretch")
        if not fresh: st.caption("No current scores — run PageRank to size nodes by rank.")

# ---------------- rankings ----------------
with tabs[2]:
    if not fresh: st.info("Run PageRank on the current network to see rankings.")
    else:
        order = np.argsort(-res.r, kind="stable")
        df = pd.DataFrame({"Rank": range(1, len(pages) + 1), "Page": [pages[i] for i in order],
                           "Score": res.r[order], "% of total": 100 * res.r[order] / res.r.sum(),
                           "In": [sum(t == pages[i] for _, t in links) for i in order],
                           "Out": [sum(s == pages[i] for s, _ in links) for i in order]})
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=float(res.r.max()), format="%.4f")})
        st.plotly_chart(viz.ranking_bar(pages, res.r), width="stretch")
        if S.base is not None and S.base[0] == pages:
            st.markdown("**Change vs. saved baseline**")
            st.dataframe(ex.compare_runs(pages, S.base[1], res), hide_index=True, width="stretch")
        if st.button("Save this run as baseline for comparison"): S.base = (list(pages), res); st.rerun()

# ---------------- math lab ----------------
with tabs[3]:
    A = adjacency_matrix(pages, links) if pages else None
    P, dang = transition_matrix(A)
    def show(m, caption):
        df = pd.DataFrame(m, index=pages, columns=pages)
        sty = df.style.format("{:.3f}")
        if S.selected: sty = sty.set_properties(subset=pd.IndexSlice[:, [S.selected]], **{"background-color": "#2b3a67"})
        st.dataframe(sty, width="stretch")
        st.caption(caption)
    st.caption("Row = target page, column = source page (column-stochastic). Selected page's column is highlighted.")
    with st.expander("1 · Adjacency matrix A"): show(A, "A[i,j]=1 when page j links to page i.")
    with st.expander("2 · Transition matrix P (after dangling fix)", expanded=True):
        show(P, "Each column is divided by the page's out-degree so it sums to 1.")
        st.write("Column sums:", np.round(P.sum(axis=0), 12).tolist())
    with st.expander("3 · Dangling-column correction"):
        if dang:
            for j in dang:
                st.write(f"**{pages[j]}** has no outgoing links → its zero column is replaced by 1/N = {1/len(pages):.4f} in every entry.")
        else: st.write("No dangling pages in this network.")
    with st.expander("4 · Damped Google matrix G = dP + (1−d)/N·11ᵀ"): show(google_matrix(P, S.d), "Strictly positive for 0 ≤ d < 1, so the stationary vector is unique.")
    if fresh:
        with st.expander("5 · Initial vector r⁽⁰⁾"): st.dataframe(pd.DataFrame({"r0": res.history[0]}, index=pages).T)
        with st.expander("6 · Intermediate vectors"):
            H = pd.DataFrame(np.array(res.history).T, index=pages, columns=[f"k={k}" for k in range(len(res.history))])
            st.dataframe(H.iloc[:, :min(12, H.shape[1])], width="stretch")
        with st.expander("7 · Convergence differences"): st.dataframe(pd.DataFrame({"iteration": range(1, res.iterations + 1), "L1 diff": res.deltas}).head(30))
        with st.expander("8 · Final verification", expanded=True):
            vr = pd.DataFrame(verify(res)); vr["status"] = vr["passed"].map({True: "✅ pass", False: "❌ fail", None: "ℹ info"})
            st.dataframe(vr[["check", "status", "value", "note"]], hide_index=True, width="stretch")
            st.caption("A large raw ‖Pr−r‖₁ is expected: the stationary equation that must hold is the *damped* one.")
    else: st.info("Run PageRank for steps 5–8.")

# ---------------- convergence ----------------
with tabs[4]:
    if not fresh: st.info("Run PageRank to see convergence.")
    else:
        m = st.columns(3); m[0].metric("Iterations", res.iterations); m[1].metric("Final L1 diff", f"{res.final_delta:.2e}")
        m[2].metric("Status", "converged" if res.converged else "limit reached")
        if not res.converged: st.error("Did not reach the tolerance within the maximum iterations.")
        st.plotly_chart(viz.convergence_figure(res), width="stretch")
        k = st.slider("Step through iteration k", 0, res.iterations, min(S.step, res.iterations))
        S.step = k
        st.plotly_chart(viz.ranking_bar(pages, res.history[k], f"Scores at iteration {k}"), width="stretch")
        sel = st.multiselect("Track pages across iterations", pages, default=pages[:3])
        if sel:
            st.line_chart(pd.DataFrame(np.array(res.history)[:, [idx[p] for p in sel]], columns=sel))
        st.caption("A shrinking difference means the scores are settling; a smaller ε keeps iterating until the change is smaller still.")

# ---------------- experiments ----------------
with tabs[5]:
    if not pages: st.stop()
    e1, e2, e3, e4, e5 = st.tabs(["1 Network effect", "2 Damping", "3 Tolerance", "4 Dangling", "5 Verification"])
    with e1:
        st.write("Edit links in Network Studio, then compare against the saved baseline (Rankings tab → *Save baseline*).")
        if fresh and S.base and S.base[0] == pages:
            cmp = ex.compare_runs(pages, S.base[1], res); st.dataframe(cmp, hide_index=True, width="stretch")
            st.write(f"Iterations: baseline {S.base[1].iterations} → current {res.iterations} · converged: {S.base[1].converged} → {res.converged}")
            st.plotly_chart(viz.grouped_scores(pages, {"Baseline": S.base[1].r, "Current": res.r}, "Score comparison"), width="stretch")
        else: st.info("Save a baseline, change a link, press Run, and return here.")
        st.caption("One link changes more than two scores because rank flows onward: a page's score is passed to everything it links to, and so on.")
    with e2:
        runs, summ, scores = ex.damping_experiment(pages, links, S.tol)
        st.dataframe(summ, hide_index=True, width="stretch")
        st.plotly_chart(viz.heatmap(pages, list(scores.columns), scores.values, "Scores by damping factor"), width="stretch")
        st.plotly_chart(viz.grouped_scores(pages, {c: scores[c].values for c in scores}, "Scores by damping factor"), width="stretch")
        st.caption("Read the table for this network only: higher d does not always mean higher scores or more iterations for every page.")
    with e3:
        runs, summ, ref = ex.tolerance_experiment(pages, links, S.d)
        st.dataframe(summ, hide_index=True, width="stretch")
        st.caption("Reference: direct solve of (I − dP)r = (1−d)/N·1 with numpy.linalg.solve, independent of the iteration.")
    with e4:
        dang_names = [pages[i] for i in range(len(pages)) if not any(s == pages[i] for s, _ in links)]
        if not dang_names: st.info("No dangling page. Load “Demo site” or “Self-link and dangling”.")
        else:
            p = st.selectbox("Dangling page", dang_names); j = idx[p]
            before = ex.run(pages, links, S.d, S.tol)
            st.write(f"**{p}** has no outgoing links. Raw column of A: `{adjacency_matrix(pages, links)[:, j].tolist()}`")
            st.write("Corrected column of P:", np.round(before.P[:, j], 4).tolist(), " (sums to 1)")
            tgt = st.selectbox(f"Add a link from {p} to", [q for q in pages if q != p] or pages)
            after = ex.run(pages, links + [(p, tgt)], S.d, S.tol)
            st.dataframe(ex.compare_runs(pages, before, after), hide_index=True, width="stretch")
            st.caption("Without the 1/N correction the column would sum to 0, P would leak probability, and scores would not sum to 1.")
    with e5:
        if fresh: st.dataframe(pd.DataFrame(verify(res)), hide_index=True, width="stretch")
        else: st.info("Run PageRank first.")
