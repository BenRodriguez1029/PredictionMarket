# Polymarket Trading System
```bash
conda create -n pmkt python=3.12 -y
conda activate pmkt
pip install -r requirements.txt
cp .env   # then fill in real credentials
```

## Layout
- `config.py` — loads secret variables from `.env`
- `db/` — connection, schema, queries
- `ingest/` — Polymarket fetch + pipeline
- `analysis/` — Data Analysis
- `notebooks/`
- `tests/`
