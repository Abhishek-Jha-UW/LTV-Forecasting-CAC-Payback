"""
OpenAI helpers: structured synthetic data and insight narratives.

Uses Streamlit secrets when available; never log raw API keys.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class SyntheticOrder(BaseModel):
    customer_id: str
    order_date: str = Field(description="ISO date YYYY-MM-DD")
    revenue: float = Field(ge=0)

    @field_validator("order_date")
    @classmethod
    def iso_date(cls, v: str) -> str:
        pd.to_datetime(v)
        return v


class SyntheticDataset(BaseModel):
    orders: list[SyntheticOrder]


class ColumnMappingSuggestion(BaseModel):
    customer_id_column: str
    order_date_column: str
    revenue_column: str
    notes: str = ""


SYNTHETIC_SYSTEM_PROMPT = """You output only valid JSON for a purchase transaction dataset.
Each order must have customer_id, order_date (YYYY-MM-DD), revenue (non-negative float).
No markdown, no commentary. Dates must fall within the user's requested window.
Create realistic repeat purchase patterns (some one-time, some loyal)."""


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


def generate_synthetic_orders(
    client,
    *,
    n_customers: int,
    start_date: str,
    end_date: str,
    industry: str,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Returns (dataframe, error_message).
    """
    user_prompt = json.dumps(
        {
            "task": "generate_orders",
            "n_customers": n_customers,
            "start_date": start_date,
            "end_date": end_date,
            "industry": industry,
            "constraints": {
                "max_orders_per_customer": 40,
                "revenue_range_hint_usd": [5, 500],
            },
        }
    )

    try:
        resp = client.chat.completions.create(
            model=openai_model(),
            temperature=0.65,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYNTHETIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        payload = json.loads(content)
        if "orders" not in payload and isinstance(payload, list):
            payload = {"orders": payload}
        dataset = SyntheticDataset.model_validate(payload)
        df = pd.DataFrame([o.model_dump() for o in dataset.orders])
        if df.empty:
            return None, "Model returned zero orders."
        df["order_date"] = pd.to_datetime(df["order_date"]).dt.normalize()
        df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
        return df, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


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
