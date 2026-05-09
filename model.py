"""
LTV forecasting, cohort views, and CAC payback utilities.

Includes:
- RFM summary and BG/NBD + Gamma-Gamma (via `lifetimes`) when data supports it
- Geometric retention / discounted cohort LTV as a transparent baseline
- Cumulative payback curves vs CAC
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize

OBSERVATION_END_COL = "_observation_end"


@dataclass
class PaybackResult:
    """Months to recover CAC (None if not within horizon)."""

    payback_month: int | None
    cumulative_margin: pd.Series
    months: np.ndarray


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], utc=True).dt.tz_convert(None)


def demo_subscription_mrr_data(
    n_customers: int = 600,
    seed: int = 7,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    monthly_churn_range: tuple[float, float] = (0.03, 0.09),
    mrr_range: tuple[float, float] = (29.0, 249.0),
) -> pd.DataFrame:
    """
    Monthly renewal rows (one per active month) — better aligned with geometric retention + CAC payback.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    rows: list[dict[str, Any]] = []

    for cid in range(n_customers):
        churn = float(rng.uniform(*monthly_churn_range))
        mrr = float(rng.uniform(*mrr_range))
        max_offset = max(1, (end_ts - start_ts).days - 60)
        first = start_ts + pd.Timedelta(days=int(rng.integers(0, max_offset)))
        first = first.normalize()
        t = first
        while t <= end_ts:
            rows.append(
                {
                    "customer_id": f"S{cid:05d}",
                    "order_date": t,
                    "revenue": round(mrr, 2),
                }
            )
            if rng.random() < churn:
                break
            t = (t + pd.DateOffset(months=1)).normalize()

    out = pd.DataFrame(rows)
    out = out.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    return out


def demo_transaction_data(
    n_customers: int = 800,
    seed: int = 42,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Synthetic repeat-purchase data (non-PII) for demos and tests.
    Mix of one-time buyers and loyal repeaters with heterogeneous spend.
    """
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    horizon_days = int((end_ts - start_ts).days)

    rows: list[dict[str, Any]] = []
    for cid in range(n_customers):
        p_repeat = float(rng.beta(2, 8))  # skew toward low repeat
        n_orders = 1
        while n_orders < 48 and rng.random() < p_repeat:
            n_orders += 1

        first_offset = int(rng.integers(0, max(1, horizon_days // 2)))
        first_date = start_ts + pd.Timedelta(days=first_offset)

        inter_purchase = rng.lognormal(mean=3.0, sigma=0.55, size=max(0, n_orders - 1))
        inter_purchase = np.maximum(inter_purchase, 1.0)

        order_dates = [first_date]
        for gap in inter_purchase:
            nxt = order_dates[-1] + pd.Timedelta(days=float(gap))
            if nxt > end_ts:
                break
            order_dates.append(nxt)

        base_spend = float(rng.lognormal(mean=3.2, sigma=0.35))
        for d in order_dates:
            noise = float(rng.lognormal(mean=0.0, sigma=0.12))
            revenue = round(base_spend * noise, 2)
            rows.append(
                {
                    "customer_id": f"C{cid:05d}",
                    "order_date": d.normalize(),
                    "revenue": revenue,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    return out


def attach_observation_end(
    transactions: pd.DataFrame,
    observation_end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Lifetimes needs a fixed observation end. Attach as constant column for clarity.
    """
    df = transactions.copy()
    if observation_end is None:
        end = pd.to_datetime(df["order_date"]).max()
    else:
        end = pd.to_datetime(observation_end)
    df[OBSERVATION_END_COL] = pd.Timestamp(end).normalize()
    return df


def transactions_to_rfm_summary(
    transactions: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "order_date",
    monetary_col: str = "revenue",
    observation_end: pd.Timestamp | str | None = None,
    freq: Literal["D", "W", "M"] = "W",
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build RFM summary compatible with `lifetimes` BG/NBD + Gamma-Gamma.
    """
    from lifetimes.utils import summary_data_from_transaction_data

    df = transactions.copy()
    df[date_col] = _ensure_datetime(df, date_col)
    if observation_end is None:
        obs_end = df[date_col].max()
    else:
        obs_end = pd.to_datetime(observation_end)

    summary = summary_data_from_transaction_data(
        df,
        customer_id_col=customer_col,
        datetime_col=date_col,
        monetary_value_col=monetary_col,
        observation_period_end=obs_end,
        freq=freq,
    )
    summary = summary.rename(
        columns={
            "frequency": "frequency",
            "recency": "recency",
            "T": "T",
            "monetary_value": "monetary_value",
        }
    )
    return summary, pd.Timestamp(obs_end)


def fit_bgf_ggf_models(
    summary: pd.DataFrame,
    bgf_penalizer: float = 0.001,
    ggf_penalizer: float = 0.01,
) -> tuple[Any, Any | None, str | None]:
    """
    Fit Beta-Geometric / NBD and Gamma-Gamma (monetary) models.
    Returns (bgf, ggf_or_none, error_message).
    """
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    if summary.empty:
        return None, None, "No customers in summary."

    bgf = BetaGeoFitter(penalizer_coef=bgf_penalizer)
    try:
        bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    except Exception as exc:  # noqa: BLE001
        return None, None, f"BG/NBD fit failed: {exc}"

    repeat = summary[summary["frequency"] > 0].copy()
    ggf: GammaGammaFitter | None = None
    err: str | None = None
    if len(repeat) < 30:
        err = "Not enough repeat purchasers for Gamma-Gamma (need ~30+)."
    else:
        ggf = GammaGammaFitter(penalizer_coef=ggf_penalizer)
        try:
            ggf.fit(repeat["frequency"], repeat["monetary_value"])
        except Exception as exc:  # noqa: BLE001
            ggf = None
            err = f"Gamma-Gamma fit failed: {exc}"

    return bgf, ggf, err


def probabilistic_ltv_table(
    summary: pd.DataFrame,
    bgf: Any,
    ggf: Any | None,
    months: int = 12,
    discount_rate_monthly: float = 0.0,
) -> pd.DataFrame:
    """
    Per-customer expected future value over `months` (converted from weekly T in lifetimes).
    If ggf is None, returns expected repeat transactions * average historical revenue proxy.
    """
    # lifetimes CLV helper expects time in same units as summary (we used weekly freq)
    time_periods = max(1, int(round(months * 30 / 7)))

    if ggf is not None:
        clv = ggf.customer_lifetime_value(
            bgf,
            summary["frequency"],
            summary["recency"],
            summary["T"],
            summary["monetary_value"],
            time=time_periods,
            discount_rate=discount_rate_monthly * (7 / 30.4375),
        )
        out = summary.copy()
        out["prob_clv"] = np.asarray(clv, dtype=float)
        return out

    expected_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
        time_periods,
        summary["frequency"],
        summary["recency"],
        summary["T"],
    )
    avg_monetary = np.where(
        summary["frequency"] > 0,
        summary["monetary_value"],
        summary["monetary_value"].replace(0, np.nan).fillna(summary["monetary_value"].mean()),
    )
    out = summary.copy()
    out["prob_clv"] = np.asarray(expected_purchases, dtype=float) * np.asarray(avg_monetary, dtype=float)
    return out


def cohort_monthly_revenue(
    transactions: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "order_date",
    revenue_col: str = "revenue",
) -> pd.DataFrame:
    """
    Cohort matrix: signup month x calendar month index -> revenue.
    Signup = first order month per customer.
    """
    df = transactions.copy()
    df[date_col] = _ensure_datetime(df, date_col)
    first = df.groupby(customer_col)[date_col].transform("min")
    df["cohort"] = first.dt.to_period("M").astype(str)
    df["order_month"] = df[date_col].dt.to_period("M").astype(str)
    df["period_index"] = (
        df[date_col].dt.to_period("M").astype("int64")
        - pd.to_datetime(first).dt.to_period("M").astype("int64")
    )

    pivot = (
        df.groupby(["cohort", "period_index"], as_index=False)[revenue_col]
        .sum()
        .sort_values(["cohort", "period_index"])
    )
    return pivot


def fit_monthly_retention_geometric(
    cohort_pivot: pd.DataFrame,
    max_relative_index: int = 18,
) -> tuple[float, pd.DataFrame]:
    """
    Estimate monthly retention from **cohort revenue decay** (not binary activity),
    which behaves more like a revenue-retention curve for payback heuristics.

    For each cohort with revenue in month 0, track mean rev[t]/rev[0] across cohorts;
    fit geometric decay r^t via least squares in log space.
    """
    wide = cohort_pivot.pivot(index="cohort", columns="period_index", values="revenue").fillna(0.0)
    cols = [c for c in wide.columns if c <= max_relative_index and c >= 0]
    wide = wide[cols]
    m0 = wide[0].replace(0, np.nan).dropna()
    if m0.empty:
        return 0.9, pd.DataFrame()

    ratios: dict[int, float] = {}
    for t in cols:
        if t == 0:
            continue
        num = wide.loc[m0.index, t]
        den = wide.loc[m0.index, 0].replace(0, np.nan)
        r = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 5:
            continue
        ratios[t] = float(np.clip(r.mean(), 1e-6, 1.0))

    if not ratios:
        return 0.9, pd.DataFrame()

    ser = pd.Series(ratios).sort_index()
    ser.index.name = "period_index"
    y = np.log(ser.values)
    x = ser.index.values.astype(float)

    def neg_sse(ret: float) -> float:
        pred = x * np.log(np.clip(ret, 1e-6, 0.999999))
        return float(np.sum((y - pred) ** 2))

    res = optimize.minimize_scalar(neg_sse, bounds=(0.55, 0.999), method="bounded")
    r_est = float(res.x) if res.success else float(np.exp((y[-1] - y[0]) / (x[-1] - x[0]) if len(x) > 1 else -0.05))
    r_est = float(np.clip(r_est, 0.55, 0.999))
    return r_est, ser.to_frame(name="mean_revenue_ratio_vs_m0")


def payback_months(
    cac: float,
    monthly_margin_per_customer: float,
    monthly_retention: float,
    max_months: int = 120,
) -> PaybackResult:
    """
    Discrete months: cumulative_margin[t] = m * sum_{k=0}^{t} r^k
    """
    r = float(monthly_retention)
    m = float(monthly_margin_per_customer)
    months = np.arange(0, max_months + 1)
    if r >= 1.0:
        cum = m * (months + 1)
    else:
        cum = m * (1.0 - r ** (months + 1)) / (1.0 - r)
    payback: int | None = None
    for t in months.astype(int):
        if cum[t] >= cac:
            payback = int(t)
            break
    series = pd.Series(cum, index=months.astype(int), name="cumulative_margin")
    return PaybackResult(payback_month=payback, cumulative_margin=series, months=months)


def scenario_grid(
    cac_values: np.ndarray,
    margin_values: np.ndarray,
    retention: float,
    max_months: int = 72,
) -> pd.DataFrame:
    records = []
    for cac in cac_values:
        for m in margin_values:
            pr = payback_months(float(cac), float(m), float(retention), max_months=max_months)
            records.append(
                {
                    "cac": float(cac),
                    "monthly_margin": float(m),
                    "retention": float(retention),
                    "payback_month": pr.payback_month,
                }
            )
    return pd.DataFrame.from_records(records)


def guess_column_mapping(columns: list[str]) -> tuple[str | None, str | None, str | None]:
    """Map common header names to (customer, date, revenue). Names must match original column strings."""
    lower_map = {str(c).strip().lower(): c for c in columns}

    def pick(keys: list[str]) -> str | None:
        for k in keys:
            if k in lower_map:
                return lower_map[k]
        return None

    cust = pick(
        [
            "customer_id",
            "cust_id",
            "user_id",
            "account_id",
            "client_id",
            "customer",
            "id",
        ]
    )
    dt = pick(
        [
            "order_date",
            "purchase_date",
            "transaction_date",
            "renewal_date",
            "invoice_date",
            "date",
            "period",
        ]
    )
    rev = pick(
        [
            "revenue",
            "amount",
            "sales",
            "total",
            "mrr",
            "arr",
            "value",
            "price",
            "payment",
        ]
    )
    return cust, dt, rev


def standardize_transaction_columns(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    revenue_col: str,
) -> pd.DataFrame:
    """Return canonical columns customer_id, order_date, revenue."""
    out = df[[customer_col, date_col, revenue_col]].copy()
    out.columns = ["customer_id", "order_date", "revenue"]
    out["order_date"] = pd.to_datetime(out["order_date"], utc=True, errors="coerce").dt.tz_convert(None)
    out["order_date"] = out["order_date"].dt.normalize()
    out["revenue"] = pd.to_numeric(out["revenue"], errors="coerce")
    out = out.dropna(subset=["customer_id", "order_date", "revenue"])
    out["customer_id"] = out["customer_id"].astype(str)
    return out.reset_index(drop=True)


def saas_ltv_steady_state(
    arpa_monthly: float,
    contribution_margin_fraction: float,
    monthly_churn: float,
) -> float | None:
    """
    Classic steady-state approximation: LTV ≈ ARPA × margin ÷ monthly_churn.
    contribution_margin_fraction is 0–1 (e.g. 0.75 for 75% CM).
    """
    if monthly_churn <= 0 or contribution_margin_fraction < 0:
        return None
    return float(arpa_monthly * contribution_margin_fraction / monthly_churn)


def estimate_arpa_and_churn_heuristic(
    df: pd.DataFrame,
    cohort_retention_monthly: float | None,
) -> tuple[float | None, float | None]:
    """
    Heuristic ARPA: average of per-customer mean revenue per row (reasonable for flat subscription MRR rows).
    Heuristic churn: 1 − r when r is cohort revenue-retention factor from fit_monthly_retention_geometric.
    """
    if df.empty or "customer_id" not in df.columns:
        return None, None
    per_cust_mean = df.groupby("customer_id", sort=False)["revenue"].mean()
    arpa = float(per_cust_mean.mean()) if len(per_cust_mean) else None
    churn: float | None = None
    if cohort_retention_monthly is not None and np.isfinite(cohort_retention_monthly):
        r = float(cohort_retention_monthly)
        if 0 < r < 1:
            churn = float(np.clip(1.0 - r, 0.001, 0.99))
    return arpa, churn


def validate_transactions_df(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    monetary_col: str,
) -> list[str]:
    issues: list[str] = []
    required = {customer_col, date_col, monetary_col}
    missing = required - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {sorted(missing)}")
        return issues
    if df.empty:
        issues.append("Dataframe is empty.")
        return issues
    try:
        _ensure_datetime(df, date_col)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"Could not parse `{date_col}` as datetimes: {exc}")
    if (df[monetary_col] < 0).any():
        issues.append("Negative revenue values detected; models assume non-negative spend.")
    return issues
