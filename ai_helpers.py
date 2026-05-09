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


SYNTHETIC_SYSTEM_PROMPT = """You output only valid JSON for a purchase transaction dataset.
Each order must have customer_id, order_date (YYYY-MM-DD), revenue (non-negative float).
No markdown, no commentary. Dates must fall within the user's requested window.
Create realistic repeat purchase patterns (some one-time, some loyal)."""


INSIGHT_SYSTEM_PROMPT = """You are a senior data scientist writing concise, accurate takeaways for revenue leaders.
Only interpret the numeric facts provided by the user JSON. Do not invent metrics or data.
Use 4-6 bullet points max, plain language, and call out uncertainty or data limits when relevant."""


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


def resolve_model(st_secrets: Any | None, default: str = "gpt-4o-mini") -> str:
    if st_secrets is None:
        return default
    try:
        m = st_secrets.get("OPENAI_MODEL") if hasattr(st_secrets, "get") else st_secrets["OPENAI_MODEL"]
        return str(m).strip() if m else default
    except Exception:  # noqa: BLE001
        return default


def generate_synthetic_orders(
    client,
    *,
    model: str,
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
            model=model,
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


def generate_insights(client, *, model: str, metrics: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Returns (markdown_text, error_message).
    """
    user_prompt = json.dumps({"computed_metrics": metrics}, default=str)
    try:
        resp = client.chat.completions.create(
            model=model,
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
