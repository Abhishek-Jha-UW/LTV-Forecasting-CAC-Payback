"""
OpenAI helpers: structured synthetic data and insight narratives.

Uses Streamlit secrets when available; never log raw API keys.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

import model as model_layer


class SyntheticOrder(BaseModel):
    customer_id: str
    order_date: str = Field(description="ISO date YYYY-MM-DD")
    revenue: float = Field(ge=0)

    @field_validator("customer_id", mode="before")
    @classmethod
    def coerce_customer_id(cls, v: Any) -> str:
        """OpenAI often emits numeric IDs in JSON; normalize to string."""
        if v is None:
            raise ValueError("customer_id is required")
        s = str(v).strip()
        if not s:
            raise ValueError("customer_id cannot be empty")
        return s

    @field_validator("revenue", mode="before")
    @classmethod
    def coerce_revenue(cls, v: Any) -> float:
        if v is None:
            raise ValueError("revenue is required")
        return float(v)

    @field_validator("order_date", mode="before")
    @classmethod
    def coerce_order_date(cls, v: Any) -> str:
        if v is None:
            raise ValueError("order_date is required")
        if hasattr(v, "isoformat"):
            return str(v.isoformat())[:10]
        s = str(v).strip()
        pd.to_datetime(s)
        return s


class SyntheticDataset(BaseModel):
    orders: list[SyntheticOrder]


class ColumnMappingSuggestion(BaseModel):
    customer_id_column: str
    order_date_column: str
    revenue_column: str
    notes: str = ""


SYNTHETIC_SYSTEM_PROMPT_ECOMMERCE = """You output only valid JSON: top-level key "orders" (array).
Each element: customer_id (string), order_date ("YYYY-MM-DD"), revenue (non-negative number).

E-commerce / retail pattern:
- Aim for roughly the requested distinct customer_id values (strings like "C0042").
- Many customers should have MULTIPLE orders across different dates (repeat buyers).
- Revenue varies by order (basket sizes differ).
- Include some one-time buyers and some loyal repeaters.
- Dates must fall strictly within the user's start_date..end_date window.
No markdown or commentary."""

SYNTHETIC_SYSTEM_PROMPT_ECOMMERCE_RETRY = """Same JSON schema as before. Your previous output was rejected for being too sparse.
Requirements:
- Average orders per customer MUST exceed 1.5.
- At least half of customers must have 2+ orders.
- Spread orders across the date range (not all on one day).
Output compact JSON only."""


INSIGHT_SYSTEM_PROMPT = """You are a senior data scientist writing concise, accurate takeaways for revenue leaders.
Only interpret the numeric facts provided by the user JSON. Do not invent metrics or data.
Use 4-6 bullet points max, plain language, and call out uncertainty or data limits when relevant."""

# Fixed for predictable cost on Streamlit Cloud / portfolio demos.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

COLUMN_MAPPING_SYSTEM_PROMPT = """You map spreadsheet columns to a transaction schema.
Reply with JSON only: keys customer_id_column, order_date_column, revenue_column, notes.
Each of the three *_column values MUST be exactly one string copied verbatim from allowed_columns (same spelling and casing).
notes is one short sentence explaining the mapping.
Do not output markdown."""


def get_openai_client(api_key: str | None):
    if not api_key:
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def resolve_openai_key(st_secrets: Any | None, environ: dict[str, str] | None = None) -> str | None:
    import os

    env = environ if environ is not None else os.environ
    if st_secrets is not None:
        try:
            key = st_secrets.get("OPENAI_API_KEY") if hasattr(st_secrets, "get") else st_secrets["OPENAI_API_KEY"]
            if key:
                return str(key).strip()
        except Exception:  # noqa: BLE001
            pass
    key = env.get("OPENAI_API_KEY")
    return key.strip() if key else None


def openai_model() -> str:
    """Always use the cheapest suitable chat model for this app."""
    return DEFAULT_OPENAI_MODEL


_SUBSCRIPTION_INDUSTRY_HINTS = (
    "saas",
    "subscription",
    "recurring",
    "mrr",
    "arr",
    "software as a service",
    "b2b software",
    "membership",
    "monthly billing",
    "annual contract",
)


def is_subscription_like_industry(industry: str) -> bool:
    """Heuristic: subscription-style businesses need dense renewal rows (LLMs rarely produce enough)."""
    s = industry.lower()
    return any(h in s for h in _SUBSCRIPTION_INDUSTRY_HINTS)


def validate_ecommerce_density(df: pd.DataFrame, n_target: int) -> tuple[bool, str]:
    """Reject sparse AI outputs that break cohort / BG-NBD demos."""
    if df.empty:
        return False, "empty dataframe"
    nu = int(df["customer_id"].nunique())
    if nu < max(5, int(n_target * 0.65)):
        return False, f"too few customers ({nu} vs target ~{n_target})"
    per = df.groupby("customer_id").size()
    mo = float(per.mean())
    if mo < 1.45:
        return False, f"mean orders/customer {mo:.2f} (need clearer repeat-purchase signal)"
    if float((per >= 2).mean()) < 0.48:
        return False, "need ≥48% of customers with 2+ orders"
    return True, "ok"


def validate_column_mapping(suggestion: ColumnMappingSuggestion, allowed: list[str]) -> str | None:
    allow = set(allowed)
    fields = (
        suggestion.customer_id_column,
        suggestion.order_date_column,
        suggestion.revenue_column,
    )
    for f in fields:
        if f not in allow:
            return f"Suggested column `{f}` is not in the uploaded file."
    if len(set(fields)) < 3:
        return "Suggested mapping repeats the same column for multiple roles."
    return None


def suggest_column_mapping(
    client,
    *,
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    industry: str,
) -> tuple[ColumnMappingSuggestion | None, str | None]:
    """
    Returns (suggestion, error). Caller must still show mapping to the user before applying.
    """
    payload = {
        "allowed_columns": columns,
        "sample_rows": sample_rows[:8],
        "industry_context": industry,
        "task": "Pick which column is customer identifier, which is transaction date, which is revenue amount.",
    }
    try:
        resp = client.chat.completions.create(
            model=openai_model(),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": COLUMN_MAPPING_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        sug = ColumnMappingSuggestion.model_validate(data)
        err = validate_column_mapping(sug, columns)
        if err:
            return None, err
        return sug, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _payload_to_orders_df(payload: dict[str, Any]) -> pd.DataFrame:
    if "orders" not in payload and isinstance(payload, list):
        payload = {"orders": payload}
    dataset = SyntheticDataset.model_validate(payload)
    df = pd.DataFrame([o.model_dump() for o in dataset.orders])
    if df.empty:
        raise ValueError("Model returned zero orders.")
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.normalize()
    return df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)


def generate_synthetic_orders(
    client: Any | None,
    *,
    n_customers: int,
    start_date: str,
    end_date: str,
    industry: str,
) -> tuple[pd.DataFrame | None, str | None, str | None]:
    """
    Returns (dataframe, error_message, info_banner).

    Subscription-like industries use the built-in renewal simulator so cohort charts stay realistic.
    E-commerce uses OpenAI with density checks, one retry, then programmatic fallback.
    """
    notice: str | None = None

    if is_subscription_like_industry(industry):
        seed = abs(hash((industry.strip().lower(), start_date, end_date, int(n_customers)))) % (2**31 - 1)
        df = model_layer.demo_subscription_mrr_data(
            n_customers=int(n_customers),
            seed=int(seed),
            start=start_date,
            end=end_date,
        )
        notice = (
            f"Subscription-style industry text detected: using the **built-in monthly renewal simulator** "
            f"({df['customer_id'].nunique():,} customers, **{len(df):,} renewal rows**) "
            "so averages and cohort curves look like real SaaS telemetry. "
            "OpenAI is skipped here (language models usually produce too few renewals per customer and hit output limits)."
        )
        return df, None, notice

    if client is None:
        return None, "Missing API key in secrets (required for non-subscription industries).", None

    base_user = {
        "task": "generate_orders",
        "n_customers": n_customers,
        "start_date": start_date,
        "end_date": end_date,
        "industry": industry,
        "constraints": {
            "customer_id_must_be_quoted_string": True,
            "min_mean_orders_per_customer": 1.5,
            "min_fraction_customers_with_repeat": 0.5,
            "max_orders_per_customer": 45,
            "revenue_range_hint_usd": [8, 750],
        },
    }

    def _call_llm(system: str, user_obj: dict[str, Any], temperature: float) -> pd.DataFrame:
        resp = client.chat.completions.create(
            model=openai_model(),
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj)},
            ],
            max_tokens=8192,
        )
        content = resp.choices[0].message.content or "{}"
        payload = json.loads(content)
        return _payload_to_orders_df(payload)

    try:
        df = _call_llm(SYNTHETIC_SYSTEM_PROMPT_ECOMMERCE, base_user, temperature=0.55)
        ok, reason = validate_ecommerce_density(df, n_customers)
        if not ok:
            df2 = _call_llm(
                SYNTHETIC_SYSTEM_PROMPT_ECOMMERCE_RETRY,
                {**base_user, "previous_rejection": reason},
                temperature=0.28,
            )
            ok2, reason2 = validate_ecommerce_density(df2, n_customers)
            if ok2:
                df = df2
                notice = (
                    f"First AI draft was sparse ({reason}); **second attempt** passed density checks "
                    f"({df['customer_id'].nunique():,} customers, mean {len(df)/df['customer_id'].nunique():.1f} orders each)."
                )
            else:
                seed = abs(hash((industry, start_date, end_date, n_customers, "fallback"))) % (2**31 - 1)
                df = model_layer.demo_transaction_data(
                    n_customers=int(n_customers),
                    seed=int(seed),
                    start=start_date,
                    end=end_date,
                )
                notice = (
                    f"AI output still looked unrealistic ({reason2}). "
                    f"Loaded **built-in ecommerce repeat-purchase simulator** instead "
                    f"({df['customer_id'].nunique():,} customers, {len(df):,} orders)."
                )
        return df, None, notice
    except ValidationError as exc:
        errs = exc.errors()[:6]
        summary = "; ".join(
            f"{'/'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in errs
        )
        n = len(exc.errors())
        seed = abs(hash((start_date, end_date, n_customers, "val_err"))) % (2**31 - 1)
        df = model_layer.demo_transaction_data(
            n_customers=int(n_customers),
            seed=int(seed),
            start=start_date,
            end=end_date,
        )
        notice = (
            f"AI JSON failed validation ({n} issue(s)): {summary}. "
            f"Using **built-in ecommerce simulator** ({len(df):,} orders)."
        )
        return df, None, notice
    except Exception as exc:  # noqa: BLE001
        seed = abs(hash((start_date, end_date, n_customers, "exc"))) % (2**31 - 1)
        df = model_layer.demo_transaction_data(
            n_customers=int(n_customers),
            seed=int(seed),
            start=start_date,
            end=end_date,
        )
        notice = f"OpenAI error ({exc}). Using built-in ecommerce simulator ({len(df):,} orders)."
        return df, None, notice


def generate_insights(client, *, metrics: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Returns (markdown_text, error_message).
    """
    user_prompt = json.dumps({"computed_metrics": metrics}, default=str)
    try:
        resp = client.chat.completions.create(
            model=openai_model(),
            temperature=0.35,
            messages=[
                {"role": "system", "content": INSIGHT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Summarize implications for growth, payback, and risk:\n" + user_prompt,
                },
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None, "Empty response from model."
        return text, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
