# Polymarket Trading System
```bash
conda create -n pred python=3.12 -y
conda activate pred
pip install -r requirements.txt
```

## Layout
- `config.py` — loads secret variables from `.env`
- `db/` — connection, schema, queries
- `ingest/` — Polymarket fetch + pipeline
- `analysis/` — Data Analysis
- `notebooks/`
- `tests/`
