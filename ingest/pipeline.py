"""Ingest orchestration + dedup. Single entry point: run_ingest(limit) (task 0.3)."""


def run_ingest(limit: int = 1000):
    raise NotImplementedError("Wire up fetch/normalize/dedup/upsert in task 0.3.")
