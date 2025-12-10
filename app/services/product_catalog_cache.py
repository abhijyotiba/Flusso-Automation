import pandas as pd
import time
import threading
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# ===============================
# CONFIG
# ===============================
# Replace with your actual Google Sheet CSV link
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1e9ppexIafl8B3qd-QOKYt0V5kQMqDJ8kyjRjUONxUYs/export?format=csv"


REFRESH_INTERVAL_SECONDS = 43200  # 12 Hours

# ===============================
# GLOBAL STATE
# ===============================
# Cache stores exact uppercase model numbers as keys
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
        # Use simple error handling for valid CSV download
        df = pd.read_csv(GOOGLE_SHEET_CSV_URL)
        
        # Clean column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        
        # Ensure we have the critical column
        if "model_no" not in df.columns:
            # Fallback: try to find a column that looks like 'model'
            found = False
            for col in df.columns:
                if "model" in col:
                    df.rename(columns={col: "model_no"}, inplace=True)
                    found = True
                    break
            if not found:
                raise ValueError("CSV missing required column 'model_no'")

        logger.info(f"[PRODUCT_CACHE] Loaded CSV with {len(df)} rows.")
        return df.fillna("")  # Replace NaNs with empty strings for safety

    except Exception as e:
        logger.error(f"[PRODUCT_CACHE] ERROR downloading Google Sheet: {e}", exc_info=True)
        raise

def _build_cache(df: pd.DataFrame):
    """Convert a DataFrame to an optimized lookup dictionary."""
    global PRODUCTS_CACHE
    logger.info("[PRODUCT_CACHE] Building in-memory dictionary...")
    
    new_cache = {}
    for _, row in df.iterrows():
        # Clean the model number key for reliable lookup
        raw_model = str(row.get("model_no", ""))
        model_key = raw_model.strip().upper()
        
        if model_key:
            # Store the full row data
            new_cache[model_key] = row.to_dict()
            
    PRODUCTS_CACHE = new_cache
    logger.info(f"[PRODUCT_CACHE] Cache ready with {len(PRODUCTS_CACHE)} products.")

def _refresh_cache():
    """Download and rebuild cache safely."""
    global LAST_REFRESH, IS_REFRESHING
    if IS_REFRESHING: return

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
        try:
            now = time.time()
            if now - LAST_REFRESH >= REFRESH_INTERVAL_SECONDS:
                logger.info("[PRODUCT_CACHE] Starting scheduled refresh...")
                _refresh_cache()
            time.sleep(60) 
        except Exception as e:
            logger.error(f"[PRODUCT_CACHE] Background loop error: {e}")
            time.sleep(60)

# ===============================
# PUBLIC API
# ===============================
def init_product_cache():
    """Initialize cache on application startup."""
    logger.info("[PRODUCT_CACHE] Initializing product catalog...")
    _refresh_cache()
    t = threading.Thread(target=_refresh_loop, daemon=True)
    t.start()

def find_products_by_model(model_query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Smart lookup that handles group numbers and variations.
    1. Checks for Exact Match.
    2. Checks for Prefix Matches (starts with).
    3. Limits results to prevent overflow.
    """
    if not model_query or not PRODUCTS_CACHE:
        return []

    target = model_query.strip().upper()
    matches = []

    # 1. Exact Match (Highest Priority)
    if target in PRODUCTS_CACHE:
        matches.append(PRODUCTS_CACHE[target])

    # 2. Prefix Match (Variations)
    # If the user searched '100.1170', we want '100.1170-PC', '100.1170-BN'
    # Iterate through keys (fast enough for <100k items in memory)
    for key, data in PRODUCTS_CACHE.items():
        if len(matches) >= limit:
            break
        
        # Check if key starts with target (and isn't the exact match we already added)
        if key.startswith(target) and key != target:
            matches.append(data)
            
    return matches