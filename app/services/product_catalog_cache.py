import pandas as pd
import time
import threading
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ===============================
# CONFIG
# ===============================
GOOGLE_SHEET_CSV_URL = (
    "YOUR_SHEET_EXPORT_LINK_HERE"  # Example: https://docs.google.com/spreadsheets/d/ID/export?format=csv
)

REFRESH_INTERVAL_SECONDS = 43200  # 12 Hour


# ===============================
# GLOBAL STATE
# ===============================
PRODUCTS_CACHE: Dict[str, Dict[str, Any]] = {}
LAST_REFRESH = 0
IS_REFRESHING = False
LOCK = threading.Lock()


# ===============================
# HELPERS
# ===============================
def _download_sheet() -> pd.DataFrame:
    """Download CSV from Google Sheets and return DataFrame."""
    logger.info("[PRODUCT_CACHE] Downloading product sheet...")

    try:
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)

        # Normalize column names for safety
        df.columns = [c.strip().lower() for c in df.columns]

        if "model_no" not in df.columns:
            raise ValueError("CSV missing required column 'model_no'")

        logger.info(f"[PRODUCT_CACHE] Loaded CSV with {len(df)} rows.")
        return df

    except Exception as e:
        logger.error(f"[PRODUCT_CACHE] ERROR downloading Google Sheet: {e}", exc_info=True)
        raise


def _build_cache(df: pd.DataFrame):
    """Convert a DataFrame to an optimized lookup dictionary."""
    global PRODUCTS_CACHE

    logger.info("[PRODUCT_CACHE] Building in-memory dictionary...")

    new_cache = {}

    for _, row in df.iterrows():
        model = str(row.get("model_no", "")).strip().upper()
        if model:
            new_cache[model] = row.to_dict()

    PRODUCTS_CACHE = new_cache
    logger.info(f"[PRODUCT_CACHE] Cache ready with {len(PRODUCTS_CACHE)} products.")


def _refresh_cache():
    """Download and rebuild cache safely."""
    global LAST_REFRESH, IS_REFRESHING

    # Prevent double refresh
    if IS_REFRESHING:
        return

    with LOCK:
        try:
            IS_REFRESHING = True
            df = _download_sheet()
            _build_cache(df)
            LAST_REFRESH = time.time()
            logger.info("[PRODUCT_CACHE] Refresh complete.")

        except Exception as e:
            logger.error(f"[PRODUCT_CACHE] Refresh failed: {e}")

        finally:
            IS_REFRESHING = False


def _refresh_loop():
    """Background thread that refreshes product data periodically."""
    logger.info("[PRODUCT_CACHE] Background refresh thread started.")

    while True:
        now = time.time()
        if now - LAST_REFRESH >= REFRESH_INTERVAL_SECONDS:
            logger.info("[PRODUCT_CACHE] Starting scheduled refresh...")
            _refresh_cache()

        time.sleep(30)  # Check every 30 seconds


# ===============================
# PUBLIC API
# ===============================
def init_product_cache():
    """Initialize cache on application startup."""
    logger.info("[PRODUCT_CACHE] Initializing product catalog...")
    _refresh_cache()

    # Start background refresh thread
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()


def get_product(model_no: str) -> Dict[str, Any]:
    """Fast lookup without network calls."""
    if not model_no:
        return None

    model_no = model_no.strip().upper()
    return PRODUCTS_CACHE.get(model_no)
