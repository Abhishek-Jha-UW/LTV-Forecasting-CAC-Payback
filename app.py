"""
LTV Forecasting & CAC Payback — Streamlit portfolio app.

Run locally:
  pip install -r requirements.txt
  streamlit run app.py

Secrets (OpenAI): copy .streamlit/secrets.toml.example → .streamlit/secrets.toml
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


def _load_uploaded_csv(uploaded) -> pd.DataFrame | None:
    if uploaded is None:
        return None
    raw = uploaded.read()
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not parse CSV: {exc}")
        return None


def _ensure_session_df() -> None:
    if "tx_df" not in st.session_state:
        st.session_state.tx_df = model.demo_subscription_mrr_data()
    if "data_profile" not in st.session_state:
        st.session_state.data_profile = "subscription_demo"


def _rfm_block(df: pd.DataFrame, freq: str) -> tuple[pd.DataFrame, Any, Any, str | None, pd.Timestamp]:
    summary, obs_end = model.transactions_to_rfm_summary(df, freq=freq)  # type: ignore[arg-type]
    bgf, ggf, err = model.fit_bgf_ggf_models(summary)
    return summary, bgf, ggf, err, obs_end


def main() -> None:
    _ensure_session_df()

    st.title("LTV forecasting & CAC payback lab")
    st.caption(
        "Cohort views, probabilistic CLV (BG/NBD + Gamma-Gamma), geometric LTV scenarios, "
        "and optional OpenAI for synthetic data + narrative insights."
    )

    with st.sidebar:
        st.header("Data")
        profile = st.radio(
            "Dataset",
            options=[
                ("subscription_demo", "Demo: subscription MRR (payback-friendly)"),
                ("ecommerce_demo", "Demo: ecommerce repeat purchases (BG/NBD)"),
                ("upload", "Upload CSV"),
                ("openai", "Generate with OpenAI"),
            ],
            format_func=lambda x: x[1],
        )
        st.session_state.data_profile = profile[0]

        if profile[0] == "subscription_demo":
            if st.button("Reset subscription demo"):
                st.session_state.tx_df = model.demo_subscription_mrr_data()
                st.rerun()
        elif profile[0] == "ecommerce_demo":
            if st.button("Reset ecommerce demo"):
                st.session_state.tx_df = model.demo_transaction_data()
                st.rerun()
        elif profile[0] == "upload":
            up = st.file_uploader("CSV: customer_id, order_date, revenue", type=["csv"])
            loaded = _load_uploaded_csv(up)
            if loaded is not None:
                st.session_state.tx_df = loaded
        else:
            st.markdown("Uses `OPENAI_API_KEY` from Streamlit secrets or env.")
            if not _openai_ready():
                st.warning("Add your key to `.streamlit/secrets.toml` (see example file).")
            n_c = st.slider("Target distinct customers", 40, 400, 120, 10)
            ind = st.text_input("Industry / context", "B2B SaaS analytics product")
            d0 = st.date_input("Start", value=pd.Timestamp("2023-01-01"))
            d1 = st.date_input("End", value=pd.Timestamp("2024-12-01"))
            if st.button("Generate dataset", type="primary"):
                key = ai_helpers.resolve_openai_key(_get_secrets())
                client = ai_helpers.get_openai_client(key)
                if client is None:
                    st.error("Missing API key.")
                else:
                    model_name = ai_helpers.resolve_model(_get_secrets())
                    df_ai, err = ai_helpers.generate_synthetic_orders(
                        client,
                        model=model_name,
                        n_customers=n_c,
                        start_date=str(d0),
                        end_date=str(d1),
                        industry=ind,
                    )
                    if err:
                        st.error(err)
                    elif df_ai is not None:
                        st.session_state.tx_df = df_ai
                        st.success(f"Generated {len(df_ai):,} orders.")
                        st.rerun()

        st.divider()
        st.header("Model settings")
        rfm_freq = st.selectbox("RFM calendar bucket", options=["W", "D", "M"], index=0, help="BG/NBD summary period.")
        horizon_m = st.slider("CLV horizon (months)", 3, 36, 12)
        discount_m = st.slider("Monthly discount rate (CLV)", 0.0, 0.02, 0.0, 0.001)

    df: pd.DataFrame = st.session_state.tx_df.copy()
    issues = model.validate_transactions_df(df, "customer_id", "order_date", "revenue")
    if issues:
        for msg in issues:
            st.error(msg)
        st.stop()

    tabs = st.tabs(["Overview", "Data & cohorts", "Probabilistic LTV", "CAC payback", "AI co-pilot"])

    cohort = model.cohort_monthly_revenue(df)

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        n_cust = df["customer_id"].nunique()
        n_ord = len(df)
        rev = df["revenue"].sum()
        aov = rev / max(1, n_ord)
        c1.metric("Customers", f"{n_cust:,}")
        c2.metric("Orders", f"{n_ord:,}")
        c3.metric("Revenue", f"${rev:,.0f}")
        c4.metric("Avg order value", f"${aov:,.2f}")

        st.subheader("Cohort revenue (index 0 = first calendar month in cohort)")
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
            "(geometric fit to mean cohort revenue vs month 0; interpret carefully for non-subscription data)."
        )
        if not curve.empty:
            fig_c = px.line(curve.reset_index(), x="period_index", y="mean_revenue_ratio_vs_m0", markers=True)
            fig_c.update_layout(yaxis_title="Mean revenue_t / revenue_0", xaxis_title="Months since first month")
            st.plotly_chart(fig_c, use_container_width=True)

    with tabs[1]:
        st.subheader("Transactions (sample)")
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download current dataset (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="ltv_cac_transactions.csv",
            mime="text/csv",
        )

        st.subheader("Long-form cohort revenue")
        st.dataframe(cohort.sort_values(["cohort", "period_index"]), use_container_width=True, height=280)

    with tabs[2]:
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
        st.subheader(f"Probabilistic CLV — next {horizon_m} months (discounted @ {discount_m:.3f}/mo)")
        st.dataframe(
            tbl_disp[["customer_id", "frequency", "recency", "T", "monetary_value", "prob_clv"]]
            .sort_values("prob_clv", ascending=False)
            .head(200),
            use_container_width=True,
            hide_index=True,
        )

        fig_v = px.histogram(tbl_disp, x="prob_clv", nbins=40, title="Distribution of per-customer CLV")
        st.plotly_chart(fig_v, use_container_width=True)

        mean_clv = float(tbl_disp["prob_clv"].mean())
        med_clv = float(tbl_disp["prob_clv"].median())
        st.metric("Mean CLV (horizon)", f"${mean_clv:,.2f}")
        st.metric("Median CLV (horizon)", f"${med_clv:,.2f}")

    with tabs[3]:
        st.markdown(
            "Geometric **margin** model: through month *t*, cumulative margin is "
            r"**Σ m·r^k** for *k* = 0…*t* (constant *m*, survival-style decay *r*<1). "
            "Payback is the first month cumulative margin crosses **CAC**."
        )
        r_auto, _ = model.fit_monthly_retention_geometric(cohort)
        c1, c2 = st.columns(2)
        with c1:
            cac = st.number_input("CAC ($)", min_value=0.0, value=350.0, step=25.0)
            margin = st.number_input("Monthly contribution margin ($)", min_value=0.0, value=95.0, step=5.0)
        with c2:
            r_eff = st.slider(
                "Monthly retention (revenue/margin)",
                min_value=0.5,
                max_value=0.999,
                value=float(np.clip(r_auto, 0.82, 0.995)) if np.isfinite(r_auto) else 0.88,
                step=0.005,
                help="Auto default clamps cohort-decay fit into a plausible SaaS band; adjust freely.",
            )
            max_m = st.slider("Max months to display", 6, 120, 48)

        pr = model.payback_months(cac, margin, r_eff, max_months=max_m)
        if pr.payback_month is None:
            st.warning("Payback not reached inside the horizon — raise margin, retention, or lower CAC.")
        else:
            st.success(f"**Payback month:** {pr.payback_month} (0-indexed from first margin month)")

        fig_p = go.Figure()
        fig_p.add_trace(
            go.Scatter(x=pr.cumulative_margin.index, y=pr.cumulative_margin.values, name="Cumulative margin")
        )
        fig_p.add_hline(y=cac, line_dash="dash", line_color="red", annotation_text="CAC")
        fig_p.update_layout(
            xaxis_title="Month",
            yaxis_title="USD",
            title="Payback curve",
            hovermode="x unified",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Scenario grid (payback month)")
        cgrid1, cgrid2 = st.columns(2)
        with cgrid1:
            cac_lo = st.number_input("CAC min", 100.0)
            cac_hi = st.number_input("CAC max", 600.0)
        with cgrid2:
            m_lo = st.number_input("Margin min", 40.0)
            m_hi = st.number_input("Margin max", 160.0)

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

    with tabs[4]:
        st.markdown(
            "The model sends **only aggregated metrics** you already computed — "
            "it should not invent numbers. Review outputs critically."
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
            metrics: dict[str, Any] = {
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
                "cohort_decay_retention_monthly_proxy": float(model.fit_monthly_retention_geometric(cohort)[0]),
            }
            st.json(metrics)

            if st.button("Generate insight memo", type="primary"):
                key = ai_helpers.resolve_openai_key(_get_secrets())
                client = ai_helpers.get_openai_client(key)
                if client:
                    with st.spinner("Calling OpenAI…"):
                        text, err = ai_helpers.generate_insights(
                            client,
                            model=ai_helpers.resolve_model(_get_secrets()),
                            metrics=metrics,
                        )
                    if err:
                        st.error(err)
                    elif text:
                        st.markdown(text)


if __name__ == "__main__":
    main()
