# Polymarket Single-Market Trading System
```bash
conda create -n pmkt python=3.12 -y
conda activate pmkt
pip install -r requirements.txt
cp .env.example .env   # then fill in real credentials
```

## Layout
- `config.py` — loads secrets from `.env`
- `db/` — connection, schema, queries
- `ingest/` — Polymarket fetch + pipeline
- `analysis/` — modules 1–7
- `notebooks/` — exploratory work
- `tests/`
