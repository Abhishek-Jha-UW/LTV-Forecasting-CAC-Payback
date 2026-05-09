"""
LTV Forecasting & CAC Payback — Streamlit portfolio app.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

OpenAI: set OPENAI_API_KEY in Streamlit secrets (Cloud) or .streamlit/secrets.toml locally.
Model is fixed to gpt-4o-mini for predictable cost.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ai_helpers
import model

# Bundled in code so Streamlit Cloud never depends on an on-disk template path.
TRANSACTION_TEMPLATE_CSV = """customer_id,order_date,revenue
C00001,2024-01-15,99.00
C00001,2024-02-15,99.00
C00001,2024-03-15,99.00
C00002,2024-01-20,149.50
C00002,2024-02-20,149.50
C00003,2024-03-01,49.99
"""

INDUSTRY_GUIDANCE: dict[str, str] = {
    "General / mixed": (
        "Use **Probabilistic CLV** when purchases are intermittent; use **SaaS benchmark** when each row is "
        "roughly recurring revenue (e.g. monthly billings)."
    ),
    "B2B SaaS (subscription)": (
        "Subscription rows fit **SaaS steady-state LTV** (ARPA × margin ÷ churn). **BG/NBD** adds insight on "
        "purchase timing but is not the same as board-level cohort NRR models."
    ),
    "E-commerce / retail": (
        "**BG/NBD + Gamma-Gamma** match repeat purchase behavior well; compare against simple averages only after "
        "checking cohort plots."
    ),
    "Automotive / big-ticket": (
        "Long gaps between purchases: probabilistic CLV is **directional**. Prefer explicit cohort cash-flow "
        "models for finance decisions."
    ),
}

METHODOLOGY = """
### What this app computes

**1. Probabilistic CLV (BG/NBD + Gamma-Gamma)**  
- Transactions are rolled into RFM-style summaries per customer (frequency, recency, age **T**, average monetary value).  
- **BG/NBD** models how many **future repeat purchases** to expect while the customer is “alive.”  
- **Gamma-Gamma** models **spend per transaction** among repeat buyers (when there are enough of them).  
- Reported **prob_clv** uses `lifetimes` discounted CLV over your horizon when Gamma-Gamma fits; otherwise  
  **expected future purchases × historical average spend** as a fallback.  
- Monthly discount rate you set is mapped into the library’s time units.

**2. Cohort revenue-decay proxy**  
- For each cohort, mean revenue in month *t* divided by mean revenue in month 0 is averaged across cohorts.  
- A geometric decay **r**^t is fit in log-space; **r** is a **heuristic** revenue-retention factor (not BG/NBD).

**3. CAC payback**  
- User supplies **CAC**, monthly **contribution margin** *m*, and retention factor **r**.  
- Cumulative margin through month *t* is Σ *m·r*^*k* for *k* = 0…*t*. Payback is the first month this meets or exceeds CAC.

**4. SaaS steady-state benchmark**  
- **LTV ≈ ARPA × contribution_margin ÷ monthly_churn** (classic steady-state; assumes constant ARPA/churn and churn ≪ 1).  
- **Fill from data** uses average per-customer mean revenue as ARPA and **1 − cohort_decay_r** as a churn proxy — rough, shown for comparison only.

---

### Industry selector

Choosing an industry **does not swap hidden formulas** for BG/NBD (that would be unsafe without validated pipelines).  
It surfaces **guidance** for how to read the same transparent outputs.
"""


st.set_page_config(
    page_title="LTV & CAC Payback Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _get_secrets() -> Any:
    try:
        return st.secrets
    except Exception:  # noqa: BLE001
        return None


def _openai_ready() -> bool:
    return ai_helpers.resolve_openai_key(_get_secrets()) is not None


def _template_csv_bytes() -> bytes:
    return TRANSACTION_TEMPLATE_CSV.encode("utf-8")


def _template_xlsx_bytes() -> bytes:
    df = pd.read_csv(io.StringIO(TRANSACTION_TEMPLATE_CSV))
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="transactions")
    return buf.getvalue()


def _load_uploaded_file(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    raw = uploaded.getvalue()
    name = (uploaded.name or "").lower()
    mime = getattr(uploaded, "type", "") or ""
    try:
        if name.endswith(".csv") or mime == "text/csv":
            return pd.read_csv(io.BytesIO(raw))
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw))
        # Fallback by sniffing
        try:
            return pd.read_csv(io.BytesIO(raw))
        except Exception:  # noqa: BLE001
            return pd.read_excel(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not parse file: {exc}")
        return None


def _init_session() -> None:
    if "tx_df" not in st.session_state:
        st.session_state.tx_df = model.demo_subscription_mrr_data()
    if "data_profile" not in st.session_state:
        st.session_state.data_profile = "subscription_demo"
    if "industry" not in st.session_state:
        st.session_state.industry = "General / mixed"
    if "upload_raw_df" not in st.session_state:
        st.session_state.upload_raw_df = None
    if "upload_hash" not in st.session_state:
        st.session_state.upload_hash = None
    if "map_customer" not in st.session_state:
        st.session_state.map_customer = "customer_id"
    if "map_date" not in st.session_state:
        st.session_state.map_date = "order_date"
    if "map_revenue" not in st.session_state:
        st.session_state.map_revenue = "revenue"


def _rfm_block(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, Any, Any, str | None, pd.Timestamp]:
    summary, obs_end = model.transactions_to_rfm_summary(df, freq=freq)  # type: ignore[arg-type]
    bgf, ggf, err = model.fit_bgf_ggf_models(summary)
    return summary, bgf, ggf, err, obs_end


def main() -> None:
    _init_session()

    st.title("LTV forecasting & CAC payback lab")
    st.caption(
        "Cohort views, probabilistic CLV (BG/NBD + Gamma-Gamma), SaaS LTV benchmark, CAC payback — "
        "optional OpenAI for synthetic data, column hints, and memos (secrets only)."
    )

    with st.sidebar:
        st.header("Context")
        _inds = list(INDUSTRY_GUIDANCE.keys())
        try:
            _iidx = _inds.index(st.session_state.industry)
        except ValueError:
            _iidx = 0
        industry = st.selectbox(
            "Industry lens (guidance only)",
            options=_inds,
            index=_iidx,
        )
        st.session_state.industry = industry
        with st.expander("How to interpret for this industry"):
            st.markdown(INDUSTRY_GUIDANCE[industry])

        st.divider()
        st.header("Data")
        _profiles = ["subscription_demo", "ecommerce_demo", "upload", "openai"]
        try:
            _pidx = _profiles.index(st.session_state.data_profile)
        except ValueError:
            _pidx = 0
        profile = st.radio(
            "Dataset",
            options=[
                ("subscription_demo", "Demo: subscription MRR"),
                ("ecommerce_demo", "Demo: ecommerce repeats"),
                ("upload", "Upload CSV / Excel"),
                ("openai", "Generate with OpenAI"),
            ],
            format_func=lambda x: x[1],
            index=_pidx,
        )
        st.session_state.data_profile = profile[0]

        if profile[0] == "subscription_demo":
            st.session_state.upload_raw_df = None
            st.session_state.upload_hash = None
            if st.button("Reset subscription demo"):
                st.session_state.tx_df = model.demo_subscription_mrr_data()
                st.rerun()
        elif profile[0] == "ecommerce_demo":
            st.session_state.upload_raw_df = None
            st.session_state.upload_hash = None
            if st.button("Reset ecommerce demo"):
                st.session_state.tx_df = model.demo_transaction_data()
                st.rerun()
        elif profile[0] == "upload":
            st.caption("Templates use columns: `customer_id`, `order_date`, `revenue`.")
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "CSV template",
                    data=_template_csv_bytes(),
                    file_name="transactions_template.csv",
                    mime="text/csv",
                    help="Example rows you can edit and re-upload.",
                )
            with c_dl2:
                st.download_button(
                    "Excel template",
                    data=_template_xlsx_bytes(),
                    file_name="transactions_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            up = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
            if up is not None:
                payload = up.getvalue()
                h = hash(payload)
                if h != st.session_state.upload_hash:
                    loaded = _load_uploaded_file(up)
                    if loaded is not None and not loaded.empty:
                        st.session_state.upload_raw_df = loaded
                        st.session_state.upload_hash = h
                        cols = list(loaded.columns)
                        gc, gd, gr = model.guess_column_mapping(cols)
                        st.session_state.map_customer = gc or cols[0]
                        st.session_state.map_date = gd or (cols[1] if len(cols) > 1 else cols[0])
                        st.session_state.map_revenue = gr or (cols[min(2, len(cols) - 1)])
                        st.rerun()

            raw = st.session_state.upload_raw_df
            if raw is not None:
                col_opts = list(raw.columns)
                st.session_state.map_customer = st.selectbox(
                    "Customer ID column",
                    options=col_opts,
                    index=col_opts.index(st.session_state.map_customer)
                    if st.session_state.map_customer in col_opts
                    else 0,
                )
                st.session_state.map_date = st.selectbox(
                    "Order / renewal date column",
                    options=col_opts,
                    index=col_opts.index(st.session_state.map_date)
                    if st.session_state.map_date in col_opts
                    else min(1, len(col_opts) - 1),
                )
                st.session_state.map_revenue = st.selectbox(
                    "Revenue / amount column",
                    options=col_opts,
                    index=col_opts.index(st.session_state.map_revenue)
                    if st.session_state.map_revenue in col_opts
                    else min(2, len(col_opts) - 1),
                )

                b_ai, b_apply = st.columns(2)
                with b_ai:
                    if st.button("Suggest mapping (AI)", disabled=not _openai_ready()):
                        client = ai_helpers.get_openai_client(ai_helpers.resolve_openai_key(_get_secrets()))
                        if client:
                            sample = raw.head(12).to_dict(orient="records")
                            sug, err = ai_helpers.suggest_column_mapping(
                                client,
                                columns=col_opts,
                                sample_rows=sample,
                                industry=st.session_state.industry,
                            )
                            if err:
                                st.error(err)
                            elif sug:
                                st.session_state.map_customer = sug.customer_id_column
                                st.session_state.map_date = sug.order_date_column
                                st.session_state.map_revenue = sug.revenue_column
                                st.info(sug.notes or "Applied AI suggestion — verify above.")
                                st.rerun()
                with b_apply:
                    apply = st.button("Apply column mapping", type="primary")

                if apply:
                    try:
                        std = model.standardize_transaction_columns(
                            raw,
                            st.session_state.map_customer,
                            st.session_state.map_date,
                            st.session_state.map_revenue,
                        )
                    except KeyError as exc:
                        st.error(f"Missing column: {exc}")
                        std = None
                    if std is not None:
                        issues = model.validate_transactions_df(std, "customer_id", "order_date", "revenue")
                        if issues:
                            for msg in issues:
                                st.error(msg)
                        else:
                            st.session_state.tx_df = std
                            st.success(f"Loaded {len(std):,} rows, {std['customer_id'].nunique():,} customers.")
                            st.rerun()
            else:
                st.info("Upload a CSV or Excel file to continue.")
        else:
            st.session_state.upload_raw_df = None
            st.session_state.upload_hash = None
            st.caption(
                "Contexts that look like **SaaS / subscriptions** use a built-in monthly renewal simulator "
                "(no API key). Other industries call **gpt-4o-mini** via secrets."
            )
            n_c = st.slider("Target distinct customers", 40, 400, 120, 10)
            ind_ctx = st.text_input("Industry / product context", "B2B SaaS analytics product")
            d0 = st.date_input("Start", value=pd.Timestamp("2023-01-01"))
            d1 = st.date_input("End", value=pd.Timestamp("2024-12-01"))
            sub_like = ai_helpers.is_subscription_like_industry(ind_ctx)
            if not _openai_ready() and not sub_like:
                st.warning("Add `OPENAI_API_KEY` in app secrets for non-subscription industries.")
            if st.button("Generate dataset", type="primary"):
                client = (
                    ai_helpers.get_openai_client(ai_helpers.resolve_openai_key(_get_secrets()))
                    if not sub_like
                    else None
                )
                if not sub_like and client is None:
                    st.error("Missing API key in secrets.")
                else:
                    df_ai, err, gen_notice = ai_helpers.generate_synthetic_orders(
                        client,
                        n_customers=n_c,
                        start_date=str(d0),
                        end_date=str(d1),
                        industry=ind_ctx,
                    )
                    if err:
                        st.error(err)
                    elif df_ai is not None:
                        if gen_notice:
                            st.info(gen_notice)
                        issues = model.validate_transactions_df(df_ai, "customer_id", "order_date", "revenue")
                        if issues:
                            for msg in issues:
                                st.error(msg)
                        else:
                            st.session_state.tx_df = df_ai
                            st.success(
                                f"Ready: **{df_ai['customer_id'].nunique():,}** customers, **{len(df_ai):,}** orders."
                            )
                            st.rerun()

        st.divider()
        st.header("Model settings")
        rfm_freq = st.selectbox(
            "RFM calendar bucket",
            options=["W", "D", "M"],
            index=0,
            help="Period for BG/NBD summary (lifetimes).",
        )
        horizon_m = st.slider("CLV horizon (months)", 3, 36, 12)
        discount_m = st.slider("Monthly discount rate (probabilistic CLV)", 0.0, 0.02, 0.0, 0.001)

    df: pd.DataFrame = st.session_state.tx_df.copy()
    issues = model.validate_transactions_df(df, "customer_id", "order_date", "revenue")
    if issues:
        for msg in issues:
            st.error(msg)
        st.stop()

    cohort = model.cohort_monthly_revenue(df)
    r_proxy, _curve_unused = model.fit_monthly_retention_geometric(cohort)

    tabs = st.tabs(
        [
            "Overview",
            "Methodology",
            "Data",
            "Probabilistic CLV",
            "SaaS benchmark",
            "CAC payback",
            "AI co-pilot",
        ]
    )

    with tabs[0]:
        st.markdown(f"**Industry lens:** {industry}")
        c1, c2, c3, c4 = st.columns(4)
        n_cust = df["customer_id"].nunique()
        n_ord = len(df)
        rev = df["revenue"].sum()
        aov = rev / max(1, n_ord)
        c1.metric("Customers", f"{n_cust:,}")
        c2.metric("Orders", f"{n_ord:,}")
        c3.metric("Revenue", f"${rev:,.0f}")
        c4.metric("Avg order value", f"${aov:,.2f}")

        st.subheader("Cohort revenue (month 0 = first calendar month in cohort)")
        heat = cohort.pivot(index="cohort", columns="period_index", values="revenue").fillna(0)
        heat = heat[[c for c in heat.columns if c <= 18]]
        fig_h = px.imshow(
            np.log1p(heat.values),
            labels=dict(x="Period", y="Cohort", color="log(1+rev)"),
            x=[str(c) for c in heat.columns],
            y=heat.index.tolist(),
            aspect="auto",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_h, use_container_width=True)

        r_est, curve = model.fit_monthly_retention_geometric(cohort)
        st.info(
            f"**Cohort revenue-decay proxy:** implied monthly factor ≈ **{r_est:.3f}** "
            "(geometric fit to mean cohort revenue vs month 0; directional only)."
        )
        if not curve.empty:
            fig_c = px.line(curve.reset_index(), x="period_index", y="mean_revenue_ratio_vs_m0", markers=True)
            fig_c.update_layout(yaxis_title="Mean revenue_t / revenue_0", xaxis_title="Months since first month")
            st.plotly_chart(fig_c, use_container_width=True)

    with tabs[1]:
        st.markdown(METHODOLOGY)

    with tabs[2]:
        st.subheader("Templates")
        t1, t2 = st.columns(2)
        with t1:
            st.download_button(
                "Download CSV template",
                data=_template_csv_bytes(),
                file_name="transactions_template.csv",
                mime="text/csv",
            )
        with t2:
            st.download_button(
                "Download Excel template",
                data=_template_xlsx_bytes(),
                file_name="transactions_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        st.subheader("Transactions (sample)")
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download current working dataset (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="ltv_cac_working_dataset.csv",
            mime="text/csv",
        )
        st.subheader("Long-form cohort revenue")
        st.dataframe(cohort.sort_values(["cohort", "period_index"]), use_container_width=True, height=280)

    with tabs[3]:
        st.markdown(
            "BG/NBD expects **repeat purchase opportunities**. Outputs are **prob_clv** over your horizon "
            f"({horizon_m} mo) with discount {discount_m:.3f}/month mapped into model time units."
        )
        with st.spinner("Fitting BG/NBD + Gamma-Gamma…"):
            summary, bgf, ggf, ggf_err, obs_end = _rfm_block(df, rfm_freq)  # type: ignore[arg-type]

        st.write(f"**Observation end:** `{obs_end.date()}` · **RFM rows:** {len(summary):,}")
        if bgf is None:
            st.error("Could not fit BG/NBD.")
            st.stop()
        if ggf_err:
            st.warning(ggf_err)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**BG/NBD parameters**")
            st.json({k: float(v) for k, v in bgf.params_.items()})
        with c2:
            if ggf is not None:
                st.markdown("**Gamma-Gamma parameters**")
                st.json({k: float(v) for k, v in ggf.params_.items()})

        tbl = model.probabilistic_ltv_table(summary, bgf, ggf, months=horizon_m, discount_rate_monthly=discount_m)
        tbl_disp = tbl.reset_index()
        if "customer_id" not in tbl_disp.columns:
            meta_cols = {"frequency", "recency", "T", "monetary_value", "prob_clv"}
            id_cols = [c for c in tbl_disp.columns if c not in meta_cols]
            if id_cols:
                tbl_disp = tbl_disp.rename(columns={id_cols[0]: "customer_id"})
        st.subheader(f"Per-customer CLV — next {horizon_m} months")
        st.dataframe(
            tbl_disp[["customer_id", "frequency", "recency", "T", "monetary_value", "prob_clv"]]
            .sort_values("prob_clv", ascending=False)
            .head(200),
            use_container_width=True,
            hide_index=True,
        )

        fig_v = px.histogram(tbl_disp, x="prob_clv", nbins=40, title="Distribution of prob_clv")
        st.plotly_chart(fig_v, use_container_width=True)

        mean_clv = float(tbl_disp["prob_clv"].mean())
        med_clv = float(tbl_disp["prob_clv"].median())
        m1, m2 = st.columns(2)
        m1.metric("Mean CLV (horizon)", f"${mean_clv:,.2f}")
        m2.metric("Median CLV (horizon)", f"${med_clv:,.2f}")

    with tabs[4]:
        st.markdown(
            "**Steady-state SaaS-style LTV** (benchmark): `LTV ≈ ARPA × contribution_margin ÷ monthly_churn`. "
            "Margin is entered as a fraction of revenue (e.g. 0.75). Not a substitute for cohort NRR modeling."
        )
        st.caption("Changed the working dataset? Click **Refresh ARPA & churn** so the heuristics match the new file.")
        arpa_default, churn_default = model.estimate_arpa_and_churn_heuristic(df, r_proxy)
        if "saas_arpa_num" not in st.session_state:
            st.session_state.saas_arpa_num = float(arpa_default or 100.0)
        if "saas_churn_num" not in st.session_state:
            st.session_state.saas_churn_num = float(churn_default or 0.05)
        if "saas_margin_slider" not in st.session_state:
            st.session_state.saas_margin_slider = 0.75

        if st.button("Refresh ARPA & churn from current data (heuristic)"):
            aa, cc = model.estimate_arpa_and_churn_heuristic(df, r_proxy)
            if aa is not None:
                st.session_state.saas_arpa_num = aa
            if cc is not None:
                st.session_state.saas_churn_num = cc
            st.rerun()

        c1, c2, c3 = st.columns(3)
        with c1:
            arpa = st.number_input(
                "ARPA (monthly $)",
                min_value=0.0,
                step=5.0,
                help="Average revenue per account per month.",
                key="saas_arpa_num",
            )
        with c2:
            margin_frac = st.slider(
                "Contribution margin (fraction)",
                0.05,
                0.95,
                key="saas_margin_slider",
            )
        with c3:
            churn_pct = st.number_input(
                "Monthly churn (fraction)",
                min_value=0.001,
                max_value=0.5,
                step=0.005,
                format="%.4f",
                key="saas_churn_num",
            )

        ltv_s = model.saas_ltv_steady_state(arpa, margin_frac, churn_pct)
        st.metric("Steady-state LTV (benchmark)", f"${ltv_s:,.2f}" if ltv_s is not None else "n/a")

        cac_cmp = st.number_input(
            "Optional CAC for LTV:CAC ($)",
            min_value=0.0,
            value=0.0,
            step=25.0,
            key="saas_cac_compare",
        )
        if cac_cmp > 0 and ltv_s is not None:
            st.metric("LTV:CAC", f"{ltv_s / cac_cmp:.2f}x")

    with tabs[5]:
        st.markdown(
            "Cumulative margin **Σ m·r^k** until it crosses **CAC** (same **r** as Overview decay proxy by default)."
        )
        r_auto, _ = model.fit_monthly_retention_geometric(cohort)
        c1, c2 = st.columns(2)
        with c1:
            cac = st.number_input("CAC ($)", min_value=0.0, value=350.0, step=25.0)
            margin = st.number_input("Monthly contribution margin ($)", min_value=0.0, value=95.0, step=5.0)
        with c2:
            r_eff = st.slider(
                "Monthly retention factor r",
                min_value=0.5,
                max_value=0.999,
                value=float(np.clip(r_auto, 0.82, 0.995)) if np.isfinite(r_auto) else 0.88,
                step=0.005,
            )
            max_m = st.slider("Max months", 6, 120, 48)

        pr = model.payback_months(cac, margin, r_eff, max_months=max_m)
        if pr.payback_month is None:
            st.warning("Payback not reached in horizon — raise margin or retention, or lower CAC.")
        else:
            st.success(f"**Payback month:** {pr.payback_month} (0 = first margin month)")

        fig_p = go.Figure()
        fig_p.add_trace(
            go.Scatter(x=pr.cumulative_margin.index, y=pr.cumulative_margin.values, name="Cumulative margin")
        )
        fig_p.add_hline(y=cac, line_dash="dash", line_color="red", annotation_text="CAC")
        fig_p.update_layout(xaxis_title="Month", yaxis_title="USD", title="Payback curve", hovermode="x unified")
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Scenario grid (payback month)")
        g1, g2 = st.columns(2)
        with g1:
            cac_lo = st.number_input("CAC min", 100.0, key="plow")
            cac_hi = st.number_input("CAC max", 600.0, key="phi")
        with g2:
            m_lo = st.number_input("Margin min", 40.0, key="mlo")
            m_hi = st.number_input("Margin max", 160.0, key="mhi")

        grid = model.scenario_grid(
            np.linspace(cac_lo, cac_hi, 8),
            np.linspace(m_lo, m_hi, 8),
            retention=r_eff,
            max_months=max_m,
        )
        pivot = grid.pivot(index="monthly_margin", columns="cac", values="payback_month")
        fig_g = px.imshow(
            pivot.values,
            x=[f"{v:.0f}" for v in pivot.columns],
            y=[f"{v:.0f}" for v in pivot.index],
            labels=dict(x="CAC", y="Monthly margin", color="Payback month"),
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
        )
        st.plotly_chart(fig_g, use_container_width=True)

    with tabs[6]:
        st.markdown(
            "Insights use **aggregated metrics only**. Synthetic data and mapping suggestions use "
            "**gpt-4o-mini** via `OPENAI_API_KEY` in secrets."
        )
        if not _openai_ready():
            st.info("Add `OPENAI_API_KEY` to Streamlit secrets to enable this tab.")
        else:
            summary_ai, bgf_ai, ggf_ai, err_ai, obs_ai = _rfm_block(df, rfm_freq)  # type: ignore[arg-type]
            tbl_ai = (
                model.probabilistic_ltv_table(
                    summary_ai,
                    bgf_ai,
                    ggf_ai,
                    months=horizon_m,
                    discount_rate_monthly=discount_m,
                )
                if bgf_ai is not None
                else None
            )
            arpa_h, churn_h = model.estimate_arpa_and_churn_heuristic(df, r_proxy)
            ltv_b = model.saas_ltv_steady_state(
                float(arpa_h or 0),
                0.75,
                float(churn_h or 0.05),
            )
            metrics: dict[str, Any] = {
                "industry_lens": industry,
                "observation_end": str(obs_ai.date()),
                "n_customers": int(df["customer_id"].nunique()),
                "n_orders": int(len(df)),
                "total_revenue": float(df["revenue"].sum()),
                "rfm_freq": rfm_freq,
                "horizon_months": horizon_m,
                "discount_rate_monthly": float(discount_m),
                "bgf_params": {k: float(v) for k, v in bgf_ai.params_.items()} if bgf_ai else None,
                "ggf_error": err_ai,
                "mean_prob_clv": float(tbl_ai["prob_clv"].mean()) if tbl_ai is not None else None,
                "median_prob_clv": float(tbl_ai["prob_clv"].median()) if tbl_ai is not None else None,
                "cohort_decay_retention_monthly_proxy": float(r_proxy)
                if np.isfinite(r_proxy)
                else None,
                "saas_benchmark_ltv_75pct_margin": float(ltv_b) if ltv_b is not None else None,
            }
            with st.expander("Payload sent to memo generator"):
                st.json(metrics)

            if st.button("Generate insight memo", type="primary"):
                client = ai_helpers.get_openai_client(ai_helpers.resolve_openai_key(_get_secrets()))
                if client:
                    with st.spinner("Calling OpenAI…"):
                        text, err = ai_helpers.generate_insights(client, metrics=metrics)
                    if err:
                        st.error(err)
                    elif text:
                        st.markdown(text)


if __name__ == "__main__":
    main()
