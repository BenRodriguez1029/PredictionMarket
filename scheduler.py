"""

scheduler.py

Runs the full ingest pipeline on a fixed interval, indefinitely.
Once every 24 hours, also runs the resolution checker to mark newly-resolved markets.
Logs every run with a timestamp so you can verify collection is working.

Usage:
    python -m scheduler                          # defaults: 15 min, limit 1000, 2000 resolutions/day
    python -m scheduler --interval 30            # every 30 minutes
    python -m scheduler --limit 500              # fetch up to 500 markets per pull
    python -m scheduler --resolution-limit 3000  # check up to 3000 markets/day for resolution

Stop with Ctrl+C.

"""

import argparse
import logging
import time
from datetime import datetime, timezone, timedelta

from ingest.pipeline import run_ingest, check_resolutions


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

def run_scheduler(
    interval_minutes: int = 15,
    limit: int = 1000,
    resolution_limit: int = 2000,
) -> None:
    """
    Call run_ingest() every interval_minutes, forever.
    Once every 24 hours, also call check_resolutions().

    Catches and logs exceptions so a single failed pull doesn't kill the loop.

    Parameters
    ----------
    interval_minutes : int
        Minutes to wait between snapshot pulls.
    limit : int
        Maximum number of markets to fetch per snapshot pull.
    resolution_limit : int
        Maximum number of markets to check for resolution per daily sweep.
    """
    interval_seconds = interval_minutes * 60
    log.info(
        f"Scheduler started — every {interval_minutes} min, limit={limit}, "
        f"resolution_limit={resolution_limit}/day"
    )

    run_number = 0
    last_resolution_check = None   # never run yet — triggers on first loop

    while True:
        run_number += 1
        log.info(f"Run #{run_number} starting")

        # Snapshot ingest
        try:
            run_ingest(limit=limit)
            log.info(f"Run #{run_number} complete")
        except Exception as e:
            log.error(f"Run #{run_number} failed: {e}", exc_info=True)

        # Daily resolution check — runs if 24h have passed since the last one
        now = datetime.now(timezone.utc)
        if last_resolution_check is None or (now - last_resolution_check) >= timedelta(hours=24):
            log.info("Running daily resolution check...")
            try:
                check_resolutions(batch_limit=resolution_limit)
                last_resolution_check = now
                log.info("Resolution check complete")
            except Exception as e:
                # Don't update last_resolution_check on failure — retry next loop
                log.error(f"Resolution check failed: {e}", exc_info=True)

        time.sleep(interval_seconds)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polymarket ingest scheduler")
    parser.add_argument(
        "--interval", type=int, default=15,
        help="Minutes between snapshot pulls (default: 15)"
    )
    parser.add_argument(
        "--limit", type=int, default=1000,
        help="Max markets per snapshot pull (default: 1000)"
    )
    parser.add_argument(
        "--resolution-limit", type=int, default=2000,
        help="Max markets to check for resolution per daily sweep (default: 2000)"
    )
    args = parser.parse_args()

    run_scheduler(
        interval_minutes=args.interval,
        limit=args.limit,
        resolution_limit=args.resolution_limit,
    )