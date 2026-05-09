# LTV Forecasting & CAC Payback Lab

Streamlit portfolio app for **probabilistic customer lifetime value (CLV)** and **CAC payback** analysis. It combines classical buy-till-you-die models (**BG/NBD + Gamma-Gamma** via [`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes)), **cohort revenue views**, **geometric margin payback** scenarios, and optional **OpenAI** flows for synthetic data and executive-style insight memos.

## Features

- **Dual demo datasets**
  - *Subscription MRR*: monthly renewal rows — aligns well with retention + payback sliders.
  - *Ecommerce repeat purchases*: irregular orders — good fit for BG/NBD / Gamma-Gamma CLV.
- **Upload your own CSV** with columns `customer_id`, `order_date`, `revenue`.
- **OpenAI (optional)**: JSON-mode synthetic order generation; insight memo from **aggregated metrics only** (no raw row dump to the model by default).
- **Secrets-ready** for [Streamlit Community Cloud](https://streamlit.io/cloud): `OPENAI_API_KEY` in app secrets.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

### OpenAI configuration

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
2. Set `OPENAI_API_KEY` (never commit `secrets.toml` — it is gitignored).
3. Optional: `OPENAI_MODEL = "gpt-4o-mini"` (or another chat model your key supports).

On Streamlit Cloud: **App settings → Secrets** and paste the same TOML.

## CSV schema

| Column        | Description                          |
|---------------|--------------------------------------|
| `customer_id` | Stable customer identifier           |
| `order_date`  | Purchase / renewal date (`YYYY-MM-DD` or ISO) |
| `revenue`     | Non-negative revenue for that order  |

## Methodology (short)

- **BG/NBD** models repeat purchase timing from RFM-style summaries; **Gamma-Gamma** models monetary value among repeat buyers. Together they support a **discounted CLV** estimate over a chosen horizon (`lifetimes` implementation).
- **CAC payback** in the UI uses a **discrete geometric sum of monthly contribution margins** until cumulative margin crosses CAC — a transparent SaaS-style baseline, not a substitute for finance-grade cohort reporting.
- **Cohort revenue-decay** retention is a **heuristic visualization** (mean cohort revenue at month *t* vs month 0). Treat it as directional unless your business matches the assumptions.

## Repository layout

| File | Role |
|------|------|
| `app.py` | Streamlit UI |
| `model.py` | Data generators, RFM, BG/NBD + GG, cohorts, payback math |
| `ai_helpers.py` | OpenAI JSON synthetic data + insight memo |
| `requirements.txt` | Pinned dependency ranges |
| `.streamlit/config.toml` | Theme / server defaults |
| `.streamlit/secrets.toml.example` | Template for secrets |

## Deploying on GitHub + Streamlit Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Cloud, **New app** → select the repo, **Main file path**: `app.py`.
3. Add **Secrets** with `OPENAI_API_KEY` if you use the AI tab (the app runs without it).

## Disclaimer

Demo and AI-generated data are **synthetic**. Models rely on assumptions (stationarity, no major mix shifts, etc.). Use for learning and portfolio demonstration — validate production decisions with your finance and data teams.

## License

Use and modify freely for your portfolio. Add a `LICENSE` file if you need explicit terms for employers or collaborators.
