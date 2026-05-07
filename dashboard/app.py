#!/usr/bin/env python3
"""Local exploratory dashboard — extend data sources via dashboard/sources_local.py."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.config import (
    ALL_PORTS_DEFAULT,
    ROOT,
    ZONE_ORDER,
    merged_data_sources,
    merged_figure_roots,
    resolve_path,
)

IMAGE_EXT = {".png", ".svg", ".jpg", ".jpeg", ".webp", ".gif"}
_HTML_EXT = {".html"}

st.set_page_config(
    page_title="Coastal exposure dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_source_cached(rel_path: str, kind: str, sample_rows: int | None) -> pd.DataFrame | None:
    path = resolve_path(rel_path)
    if not path.is_file():
        return None
    if kind == "parquet":
        df = pd.read_parquet(path)
        if sample_rows is not None and len(df) > sample_rows:
            df = df.sample(n=min(sample_rows, len(df)), random_state=42)
        return df
    return pd.read_csv(path)


def load_source(spec: dict) -> tuple[pd.DataFrame | None, Path]:
    path = resolve_path(spec["rel_path"])
    sr = spec.get("sample_rows")
    sr_int = int(sr) if sr is not None else None
    try:
        df = load_source_cached(spec["rel_path"], spec.get("kind", "csv"), sr_int)
    except Exception:
        return None, path
    return df, path


def filter_ports(df: pd.DataFrame | None, col: str | None, ports: list[str]) -> pd.DataFrame | None:
    if df is None or not ports or col is None or col not in df.columns:
        return df
    return df.loc[df[col].astype(str).isin(ports)].copy()


def zone_ordered_series(s: pd.Series) -> tuple[pd.Series, list[str]]:
    cats = ZONE_ORDER + [str(z) for z in s.dropna().astype(str).unique() if z not in ZONE_ORDER]
    return pd.Categorical(s, categories=cats, ordered=True), cats


def collect_media(port_slugs_for_gallery: tuple[str, ...]) -> tuple[list[Path], list[Path]]:
    imgs: list[Path] = []
    html_docs: list[Path] = []
    rel_slugs = tuple(s.lower() for s in port_slugs_for_gallery)
    neutral = ("turku", "mariehamn", "stockholm")

    def keep(rel_posix_low: str) -> bool:
        if not port_slugs_for_gallery:
            return True
        if any(sl in rel_posix_low for sl in rel_slugs):
            return True
        return not any(n in rel_posix_low for n in neutral)

    for rooted in merged_figure_roots():
        rp = ROOT / rooted
        if not rp.is_dir():
            continue
        for p in sorted(rp.rglob("*")):
            if not p.is_file():
                continue
            rel_low = p.relative_to(ROOT).as_posix().lower()
            if not keep(rel_low):
                continue
            suf = p.suffix.lower()
            if suf in IMAGE_EXT:
                imgs.append(p)
            elif suf in _HTML_EXT:
                html_docs.append(p)

    key = lambda x: x.relative_to(ROOT).as_posix().lower()
    return sorted(imgs, key=key), sorted(html_docs, key=key)


def decay_plot(sl: pd.DataFrame, metric: str, wind_choice: str) -> None:
    slab = sl[sl["metric"].astype(str) == metric].copy()
    if wind_choice == "(use wind_regime = all)":
        slab = slab[slab["wind_regime"].astype(str) == "all"]
    else:
        slab = slab[slab["wind_regime"].astype(str) == wind_choice]
    slab = slab[slab["distance_zone"].notna()]
    if slab.empty:
        st.warning("No rows.")
        return
    _, cats = zone_ordered_series(slab["distance_zone"])
    slab["distance_zone_ord"] = pd.Categorical(slab["distance_zone"], categories=cats, ordered=True)
    slab = slab.sort_values(["distance_zone_ord", "port"])

    fig = px.line(
        slab,
        x="distance_zone",
        y="mean",
        color="port",
        markers=True,
        category_orders={"distance_zone": cats},
        title=f"{metric}" + (" — " + wind_choice if wind_choice != "(use wind_regime = all)" else " — pooled (wind_regime=all)"),
    )
    fig.update_layout(xaxis_tickangle=-25)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    pivot = slab.pivot_table(
        index=["distance_zone", "wind_regime"],
        columns="port",
        values=["mean", "n"],
        aggfunc="first",
    )
    with st.expander("Underlying table snippet"):
        st.dataframe(pivot, use_container_width=True)


def main() -> None:
    registry = merged_data_sources()

    st.sidebar.title("Coastal dashboard")
    st.sidebar.caption(str(ROOT))

    preset = st.sidebar.radio(
        "Port focal set",
        ("Turku + Mariehamn", "Turku only", "Mariehamn only", "Include Stockholm etc.", "Custom"),
        index=0,
    )
    if preset == "Turku + Mariehamn":
        ports = ["Turku", "Mariehamn"]
    elif preset == "Turku only":
        ports = ["Turku"]
    elif preset == "Mariehamn only":
        ports = ["Mariehamn"]
    elif preset == "Include Stockholm etc.":
        ports = list(ALL_PORTS_DEFAULT)
    else:
        ports = st.sidebar.multiselect("Ports", ALL_PORTS_DEFAULT, default=["Turku", "Mariehamn"])
        if not ports:
            ports = ["Turku", "Mariehamn"]

    gallery_mode = st.sidebar.radio(
        "Figure gallery dataset",
        (
            "Filter: selected ports + neutral figures",
            "Show entire outputs tree (PNG/HTML)",
        ),
        index=0,
    )
    gallery_slugs = () if gallery_mode.startswith("Show entire") else tuple(ports)

    imgs, htmls = collect_media(gallery_slugs)

    decay_spec = next((s for s in registry if s["id"] == "port_decay"), None)
    thesis_decay_spec = next((s for s in registry if s["id"] == "thesis_decay_tm"), None)

    tab_home, tab_decay, tab_lag, tab_grid, tab_gallery, tab_registry = st.tabs(
        ["Home", "Distance decay", "Temporal / anomalies", "Grid panel sample", "Media gallery", "Data registry"],
    )

    with tab_home:
        st.header("Integrated view")
        st.markdown(
            "Use the sidebar **Port focal set** to switch Turku-, Mariehamn-, or comparative context. "
            "New CSV/Parquet outputs can be registered in **`dashboard/sources_local.py`**.",
        )
        rspec = next((s for s in registry if s["id"] == "port_ranking"), None)
        if rspec:
            df, rp = load_source(rspec)
            df = filter_ports(df, rspec.get("port_column"), ports)
            st.subheader("Exposure ranking snapshot")
            if df is None:
                st.error(f"Missing or unreadable `{rp.relative_to(ROOT)}`")
            elif df.empty:
                st.warning("No ranking rows after port filter.")
            else:
                st.dataframe(df, use_container_width=True, hide_index=True)
                numeric = df.select_dtypes(include="number").columns.tolist()
                if numeric:
                    tidy = pd.melt(
                        df,
                        id_vars=[c for c in df.columns if c not in numeric],
                        value_vars=numeric,
                        var_name="metric",
                        value_name="value",
                    )
                    if "port" in tidy.columns:
                        fig_r = px.bar(
                            tidy,
                            x="metric",
                            y="value",
                            color="port",
                            barmode="group",
                            title="Numeric ranking fields — compare focal ports",
                        )
                        fig_r.update_layout(xaxis_tickangle=-40)
                        st.plotly_chart(fig_r, use_container_width=True)

    with tab_decay:
        st.header("Distance decay explorer")
        which = st.radio(
            "Table",
            (
                "Full pipeline decay",
                "Thesis bundle (Turku–Mariehamn)",
            ),
            horizontal=True,
        )
        spec = decay_spec if which.startswith("Full") else thesis_decay_spec
        if not spec:
            st.error("Decay source not configured.")
        else:
            df, rp = load_source(spec)
            df = filter_ports(df, spec.get("port_column"), ports)
            st.caption(str(rp.relative_to(ROOT)))
            if df is None:
                st.error(f"File missing `{rp.relative_to(ROOT)}` — run upstream analysis.")
            elif df.empty:
                st.warning("Nothing to plot after port filter.")
            else:
                metrics = sorted(df["metric"].dropna().astype(str).unique())
                regimes = sorted(df["wind_regime"].dropna().astype(str).unique())
                c1, c2 = st.columns([2, 1])
                with c1:
                    ix = (
                        metrics.index("maritime_exposure_index")
                        if "maritime_exposure_index" in metrics
                        else min(9, len(metrics) - 1)
                    )
                    metric_pick = st.selectbox("Indicator (metric)", metrics, index=ix)
                with c2:
                    reg_opts = ["(use wind_regime = all)"] + [r for r in regimes if r != "all"]
                    wind_pick = st.selectbox("Wind stratum", reg_opts)

                decay_plot(df, metric_pick, wind_pick)

    with tab_lag:
        st.header("Lag & anomaly tables (thesis)")
        lag_spec = next((s for s in registry if s["id"] == "lag_relationships"), None)
        ano_spec = next((s for s in registry if s["id"] == "anomaly_overlap"), None)

        if lag_spec:
            ldf, lp = load_source(lag_spec)
            st.subheader("Lag correlations")
            st.caption(str(lp.relative_to(ROOT)))
            if ldf is None:
                st.error("Lag table missing.")
            else:
                block_f = st.multiselect("Block filter", sorted(ldf["block"].dropna().unique()))
                sdf = ldf[ldf["block"].isin(block_f)] if block_f else ldf.copy()
                st.dataframe(sdf.head(250), use_container_width=True)
                sdf2 = sdf.copy()
                sdf2["_pair"] = sdf2["source_feature"].astype(str) + " → " + sdf2["target_feature"].astype(str)
                if "spearman_r" in sdf2.columns:
                    piv = sdf2.pivot_table(
                        index="_pair",
                        columns="lag",
                        values="spearman_r",
                        aggfunc="mean",
                    )
                    fig_h = px.imshow(
                        piv,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        zmin=-1,
                        zmax=1,
                        title="Lag heatmap — Spearman ρ (pooled; methodology in thesis reports)",
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

        if ano_spec:
            adf, ap = load_source(ano_spec)
            st.subheader("Anomaly overlap")
            st.caption(str(ap.relative_to(ROOT)))
            if adf is None:
                st.warning("Anomaly table missing.")
            else:
                adf_f = filter_ports(adf.copy(), ano_spec.get("port_column"), ports)
                st.dataframe(adf_f.head(400), use_container_width=True)
                if {"port", "anomaly_type"}.issubset(adf_f.columns):
                    ct = adf_f.groupby(["port", "anomaly_type"]).size().reset_index(name="n")
                    fig_a = px.bar(
                        ct,
                        x="anomaly_type",
                        y="n",
                        color="port",
                        barmode="group",
                        title="Anomaly flags by type × port (row counts)",
                    )
                    fig_a.update_layout(xaxis_tickangle=-35)
                    st.plotly_chart(fig_a, use_container_width=True)

    with tab_grid:
        st.header("Weekly grid sample")
        pspec = next((s for s in registry if s["id"] == "features_panel"), None)
        if pspec:
            gf, gp = load_source(pspec)
            st.caption(str(gp.relative_to(ROOT)))
            if gf is None:
                st.warning("Parquet not found — ingest / build panel first.")
            else:
                gf2 = (
                    filter_ports(gf, pspec.get("port_column"), ports)
                    if pspec.get("port_column") in gf.columns
                    else gf.copy()
                )
                cols_default = [
                    "nearest_port",
                    "week_start_utc",
                    "vessel_density_t",
                    "no2_mean_t",
                    "maritime_exposure_index",
                    "grid_centroid_lat",
                    "grid_centroid_lon",
                ]
                avail = [c for c in cols_default if c in gf2.columns]
                extra = [c for c in gf2.columns if c not in avail]
                chosen = st.multiselect(
                    "Columns",
                    gf2.columns.tolist(),
                    default=[c for c in avail if c in gf2.columns],
                )
                if not chosen:
                    chosen = sorted(gf2.columns.tolist())[: min(12, len(gf2.columns))]
                disp = gf2[chosen].copy()
                st.metric("Sample rows in view", len(disp))
                st.dataframe(disp.head(500), use_container_width=True)
        else:
            st.info("No parquet source registered.")

    with tab_gallery:
        st.header("Figures & HTML maps")
        st.caption(f"{len(imgs)} images, {len(htmls)} HTML under configured roots.")
        sub = st.radio("Layout", ("Thumbnails grid", "Pick one file"), horizontal=True)
        if sub == "Pick one file":
            label_to_path = {p.relative_to(ROOT).as_posix(): p for p in imgs}
            if not label_to_path:
                st.info("No PNG/SVG files found under figure roots.")
            else:
                choice = st.selectbox("Image", sorted(label_to_path.keys()))
                st.image(str(label_to_path[choice]), use_container_width=True)
        else:
            step = 3
            imgs_cap = imgs[: 60 * step]
            for i in range(0, len(imgs_cap), step):
                row = imgs_cap[i : i + step]
                cols = st.columns(len(row))
                for c2, pimg in zip(cols, row):
                    with c2:
                        st.markdown(f"**/{pimg.parent.name}/{pimg.name}**")
                        st.image(str(pimg), use_container_width=True)
            if len(imgs) > len(imgs_cap):
                st.warning("Showing first thumbnails only — use Pick one file for the rest.")

        if htmls:
            st.subheader("Interactive HTML")
            h_choice = st.selectbox(
                "Open map / visualization",
                [h.relative_to(ROOT).as_posix() for h in htmls],
                key="which_html_gallery_doc",
            )
            hpath = ROOT / h_choice
            html_content = hpath.read_text(encoding="utf-8", errors="replace")
            st.components.v1.html(html_content, height=600, scrolling=True)

    with tab_registry:
        st.header("Registered data sources")
        meta = []
        for s in registry:
            rp = resolve_path(s["rel_path"])
            meta.append(
                {
                    "id": s["id"],
                    "label": s.get("label", ""),
                    "category": s.get("category", ""),
                    "exists": rp.is_file(),
                    "path": str(rp.relative_to(ROOT)),
                    "port_column": s.get("port_column"),
                    "kind": s.get("kind", "csv"),
                },
            )
        st.dataframe(pd.DataFrame(meta), use_container_width=True, hide_index=True)

        findings = next((s for s in registry if s["id"] == "key_findings"), None)
        if findings:
            kdf, _kp = load_source(findings)
            if kdf is not None:
                st.subheader(findings.get("label", "Key findings"))
                st.dataframe(kdf, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
