import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. Add it to your .env file."
        )
    return val


# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = _require("DB_NAME")
DB_USER = _require("DB_USER")
DB_PASSWORD = _require("DB_PASSWORD")

# Spread tiers for categorizing market liquidity based on relative spread
LIQUIDITY_TIERS = {
    "tight":       0.02,
    "medium":      0.05,
    "wide":        0.10,
    # anything above the last threshold falls into "untradeable"
}

# Filters
MIN_VOLUME_24H = 1000.0

# Resolution
RESOLUTION_THRESHOLD = 0.99           # winning side >= this; losing side <= 1 - this
RESOLUTION_SETTLE_BUFFER_HOURS = 48
RESOLUTION_MAX_AGE_DAYS = 90

# Data Calibration
OBSERVATION_HORIZON_DAYS = 7      # observe price this many days before resolution
OBSERVATION_TOLERANCE_HOURS = 12     # accept a snapshot within +/- this many hours of the target
