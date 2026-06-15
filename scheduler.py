"""

scheduler.py

Runs the full ingest pipeline on a fixed interval, indefinitely.
Logs every run with a timestamp so you can verify collection is working.

Usage:
    python -m scheduler                    # defaults: every 15 min, 1000 markets
    python -m scheduler --interval 30      # every 30 minutes
    python -m scheduler --limit 500        # fetch up to 500 markets per run

Stop with Ctrl+C.

"""

import argparse
import logging
import time
from datetime import datetime, timezone

from ingest.pipeline import run_ingest


# Logging — writes to console and to logs/scheduler.log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                        # console
        logging.FileHandler("logs/scheduler.log"),      # file
    ],
)
log = logging.getLogger(__name__)


# Scheduler

def run_scheduler(interval_minutes: int = 15, limit: int = 1000) -> None:
    """
    Call run_ingest() every interval_minutes, forever.

    Catches and logs exceptions so a single failed pull doesn't kill the loop.
    Logs the next scheduled run time after each pull so you can verify timing.

    Parameters
    ----------
    interval_minutes : int
        How many minutes to wait between pulls.
    limit : int
        Maximum number of markets to fetch per pull.
    """
    interval_seconds = interval_minutes * 60
    log.info(f"Scheduler started — every {interval_minutes} min, limit={limit}")

    run_number = 0

    while True:
        run_number += 1
        log.info(f"Run #{run_number} starting")

        try:
            run_ingest(limit=limit)
            log.info(f"Run #{run_number} complete")
        except Exception as e:
            # Log the failure but keep the loop alive
            log.error(f"Run #{run_number} failed: {e}", exc_info=True)

        # Log when the next run will happen
        next_run = datetime.now(timezone.utc)
        log.info(
            f"Next run in {interval_minutes} min "
            f"(~{next_run.strftime('%H:%M')} UTC + {interval_minutes}m)"
        )

        time.sleep(interval_seconds)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket ingest scheduler")
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Minutes between pulls (default: 15)"
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Max markets per pull (default: 1000)"
    )
    args = parser.parse_args()

    run_scheduler(interval_minutes=args.interval, limit=args.limit)