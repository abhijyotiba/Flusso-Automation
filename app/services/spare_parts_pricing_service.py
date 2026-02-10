"""
Spare Parts Pricing Service
Downloads pricing data from Google Drive (via Service Account) and provides fast lookup.

Features:
- Google Drive API with Service Account authentication
- Multiple search strategies (exact, prefix, base model, fuzzy)
- Part number normalization for reliable matching
- Automatic refresh every 24 hours
- Handles "$ -" as "price not set" (not "doesn't exist")

Data Source: Google Drive spreadsheet (Spare-Part-Pricing)
"""

import io
import logging
import os
import re
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import settings
try:
    from app.config.settings import settings
    SPARE_PARTS_SHEET_FILE_ID = settings.spare_parts_sheet_file_id or os.getenv("SPARE_PARTS_SHEET_FILE_ID", "")
    REFRESH_INTERVAL_SECONDS = settings.spare_parts_refresh_hours * 3600  # Convert hours to seconds
except ImportError:
    SPARE_PARTS_SHEET_FILE_ID = os.getenv("SPARE_PARTS_SHEET_FILE_ID", "")
    REFRESH_INTERVAL_SECONDS = 86400  # 24 hours default

# Search configuration
FUZZY_MATCH_THRESHOLD = 0.80  # Minimum similarity for fuzzy matches
MAX_FUZZY_CANDIDATES = 50     # Max candidates to check for fuzzy match

# Finish code suffixes (for stripping to find base model)
FINISH_CODES = {"CP", "BN", "PN", "MB", "SB", "BB", "SS", "BG", "PS", "GW", "GB", "RB"}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SparePartInfo:
    """Represents a spare part with pricing information."""
    part_number: str           # Original part number from sheet
    normalized_number: str     # Normalized for lookup
    price_raw: str             # Raw price string (e.g., "$24.00" or "$ -")
    price_numeric: Optional[float]  # Parsed numeric price or None
    has_price: bool            # True if actual price exists
    is_obsolete: bool          # True if marked as obsolete
    is_display_dummy: bool     # True if marked as display dummy
    base_model: str            # Base model without finish code


# =============================================================================
# GOOGLE DRIVE API CLIENT
# =============================================================================

def _get_drive_service():
    """
    Create Google Drive API service using Service Account credentials.
    
    Credentials can be provided via:
    1. GOOGLE_APPLICATION_CREDENTIALS env var (path to JSON key file)
    2. GOOGLE_CLOUD_PROJECT + default credentials (for Cloud Run)
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        
        # Try to get credentials
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if creds_path and os.path.exists(creds_path):
            # Local development: use service account key file
            logger.info(f"[SPARE_PARTS] Using service account from: {creds_path}")
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
        else:
            # Cloud environment: use default credentials
            import google.auth
            logger.info("[SPARE_PARTS] Using default Google Cloud credentials")
            credentials, project = google.auth.default(
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
        
        service = build('drive', 'v3', credentials=credentials)
        return service
        
    except ImportError as e:
        logger.warning(f"[SPARE_PARTS] Google API libraries not installed: {e}")
        logger.info("[SPARE_PARTS] Install with: pip install google-api-python-client google-auth")
        logger.info("[SPARE_PARTS] Will use local CSV fallback instead.")
        return None
    except Exception as e:
        error_msg = str(e)
        if "credentials were not found" in error_msg.lower() or "could not automatically determine" in error_msg.lower():
            logger.warning(
                f"[SPARE_PARTS] ⚠️ Google credentials not configured.\n"
                f"  → Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file path.\n"
                f"  → Will use local CSV fallback instead."
            )
        else:
            logger.warning(f"[SPARE_PARTS] ⚠️ Failed to initialize Google Drive service: {error_msg[:150]}")
            logger.info("[SPARE_PARTS] Will use local CSV fallback instead.")
        return None


def _download_sheet_from_drive(file_id: str) -> Optional[pd.DataFrame]:
    """
    Download spreadsheet from Google Drive as CSV and return DataFrame.
    """
    if not file_id:
        logger.error("[SPARE_PARTS] No SPARE_PARTS_SHEET_FILE_ID configured")
        return None
    
    service = _get_drive_service()
    if not service:
        return None
    
    try:
        from googleapiclient.http import MediaIoBaseDownload
        
        logger.info(f"[SPARE_PARTS] Downloading sheet from Google Drive (ID: {file_id[:10]}...)")
        
        # Export as CSV
        request = service.files().export_media(
            fileId=file_id,
            mimeType='text/csv'
        )
        
        # Download to memory
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.debug(f"[SPARE_PARTS] Download progress: {int(status.progress() * 100)}%")
        
        # Reset buffer position
        file_buffer.seek(0)
        
        # Parse CSV
        df = pd.read_csv(file_buffer)
        logger.info(f"[SPARE_PARTS] Downloaded {len(df)} rows from Google Drive")
        
        return df
    
    except Exception as e:
        # Handle specific Google API errors with user-friendly messages
        error_msg = str(e)
        
        # Check for common error types
        if "HttpError 404" in error_msg or "File not found" in error_msg:
            logger.warning(
                f"[SPARE_PARTS] ⚠️ Google Drive file not accessible (404 Not Found).\n"
                f"  → File ID: {file_id}\n"
                f"  → This usually means:\n"
                f"    1. The file doesn't exist, OR\n"
                f"    2. The service account doesn't have access to this file.\n"
                f"  → Solution: Share the Google Sheet with the service account email address.\n"
                f"  → Falling back to local CSV file..."
            )
        elif "HttpError 403" in error_msg or "forbidden" in error_msg.lower():
            logger.warning(
                f"[SPARE_PARTS] ⚠️ Google Drive access forbidden (403).\n"
                f"  → The service account lacks permission to access this file.\n"
                f"  → Solution: Share the Google Sheet with the service account email (with Viewer access).\n"
                f"  → Falling back to local CSV file..."
            )
        elif "HttpError 401" in error_msg or "unauthorized" in error_msg.lower():
            logger.warning(
                f"[SPARE_PARTS] ⚠️ Google Drive authentication failed (401).\n"
                f"  → The service account credentials may be invalid or expired.\n"
                f"  → Solution: Check the service account JSON file and regenerate if needed.\n"
                f"  → Falling back to local CSV file..."
            )
        else:
            # Generic error - log without full traceback
            logger.warning(f"[SPARE_PARTS] ⚠️ Google Drive download failed: {error_msg[:200]}")
            logger.warning(f"[SPARE_PARTS] Falling back to local CSV file...")
        
        return None


def _load_local_csv_fallback() -> Optional[pd.DataFrame]:
    """
    Fallback: Load from local file if Drive access fails.
    Supports both CSV and XLSX formats.
    Useful for local development and testing.
    
    Priority order:
    1. data/spare_parts_pricing.csv (preferred - standard location)
    2. data/spare_parts_pricing.xlsx
    3. data/Spare-Part-Pricing - Sheet1.csv (legacy name)
    4. Spare-Part-Pricing - Sheet1.csv (root folder - for backwards compatibility)
    """
    local_paths = [
        # Standard location in data folder (preferred)
        ("data/spare_parts_pricing.csv", "csv"),
        ("data/spare_parts_pricing.xlsx", "xlsx"),
        # Legacy/alternate names in data folder
        ("data/Spare-Part-Pricing - Sheet1.csv", "csv"),
        ("data/Spare-Part-Pricing.csv", "csv"),
        ("data/Spare-Part-Pricing.xlsx", "xlsx"),
        # Root folder fallback (backwards compatibility)
        ("Spare-Part-Pricing - Sheet1.csv", "csv"),
        ("Spare-Part-Pricing.csv", "csv"),
    ]
    
    for path, file_type in local_paths:
        if os.path.exists(path):
            logger.info(f"[SPARE_PARTS] Loading from local file: {path}")
            try:
                if file_type == "xlsx":
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path)
                logger.info(f"[SPARE_PARTS] ✅ Loaded {len(df)} rows from local {file_type.upper()}")
                return df
            except Exception as e:
                logger.error(f"[SPARE_PARTS] Failed to load {path}: {e}")
    
    logger.error("[SPARE_PARTS] ❌ No local spare parts file found! Expected: data/spare_parts_pricing.csv")
    return None


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def normalize_part_number(raw: str) -> str:
    """
    Normalize part number for consistent lookup.
    
    Transformations:
    - Uppercase
    - Remove extra spaces
    - Standardize separators (keep dots and dashes)
    - Handle common variations
    """
    if not raw:
        return ""
    
    normalized = raw.strip().upper()
    
    # Remove multiple spaces
    normalized = re.sub(r'\s+', '', normalized)
    
    # Standardize: Replace multiple dots/dashes with single
    normalized = re.sub(r'\.+', '.', normalized)
    normalized = re.sub(r'-+', '-', normalized)
    
    return normalized


def extract_base_model(part_number: str) -> str:
    """
    Extract base model by removing finish code suffix.
    
    Examples:
    - "TVH.5007" → "TVH.5007" (no finish code)
    - "100.1800-2353CP" → "100.1800-2353"
    - "TRM.TVH.4511CP" → "TRM.TVH.4511"
    """
    if not part_number:
        return ""
    
    # Check if ends with a known finish code (2 characters)
    for finish_code in FINISH_CODES:
        if part_number.endswith(finish_code):
            return part_number[:-len(finish_code)]
    
    return part_number


def parse_price(price_str: str) -> Tuple[Optional[float], bool]:
    """
    Parse price string and return (numeric_value, has_price).
    
    Examples:
    - "$24.00" → (24.0, True)
    - "$ -" → (None, False)
    - "$0.00" → (0.0, True)  # Explicit zero is still a price
    """
    if not price_str or not isinstance(price_str, str):
        return None, False
    
    price_str = price_str.strip()
    
    # Check for "$ -" pattern (price not set)
    if price_str == "$ -" or price_str == "$-" or price_str == "-":
        return None, False
    
    # Extract numeric value
    try:
        # Remove $ and any other non-numeric chars except decimal
        numeric_str = re.sub(r'[^\d.]', '', price_str)
        if numeric_str:
            value = float(numeric_str)
            return value, True
        return None, False
    except (ValueError, TypeError):
        return None, False


def is_obsolete_part(part_number: str) -> bool:
    """Check if part is marked as obsolete."""
    upper = part_number.upper()
    return "OBSOLETE" in upper or "OBSLETE" in upper  # Handle typo in data


def is_display_dummy(part_number: str) -> bool:
    """Check if part is a display dummy (not for sale)."""
    upper = part_number.upper()
    return "DISPLAY-DUMMY" in upper or "DISPLAY DUMMY" in upper or "-DUMMY" in upper


# =============================================================================
# SPARE PARTS CACHE
# =============================================================================

class SparePartsPricingCache:
    """In-memory cache for spare parts pricing with multiple search indexes."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Primary data storage
        self.parts: Dict[str, SparePartInfo] = {}  # normalized_number → SparePartInfo
        
        # Search indexes
        self.exact_index: Dict[str, SparePartInfo] = {}       # Exact normalized match
        self.prefix_index: Dict[str, List[SparePartInfo]] = {}  # First 6 chars → parts
        self.base_model_index: Dict[str, List[SparePartInfo]] = {}  # Base model → all finishes
        
        # Refresh tracking
        self.last_refresh: float = 0
        self.is_refreshing: bool = False
        self.part_count: int = 0
        
        # Background refresh thread
        self._refresh_thread: Optional[threading.Thread] = None
    
    def refresh(self, force: bool = False) -> bool:
        """
        Download and rebuild the cache.
        
        Args:
            force: If True, refresh even if recently refreshed
            
        Returns:
            True if refresh succeeded
        """
        if self.is_refreshing and not force:
            logger.info("[SPARE_PARTS] Refresh already in progress, skipping")
            return False
        
        # Check if refresh is needed
        now = time.time()
        if not force and (now - self.last_refresh) < REFRESH_INTERVAL_SECONDS:
            logger.debug("[SPARE_PARTS] Cache still fresh, skipping refresh")
            return True
        
        with self._lock:
            self.is_refreshing = True
            
            try:
                # Try Google Drive first
                df = _download_sheet_from_drive(SPARE_PARTS_SHEET_FILE_ID)
                
                # Fallback to local CSV
                if df is None:
                    # Note: Drive error message already logged by _download_sheet_from_drive
                    df = _load_local_csv_fallback()
                
                if df is None:
                    logger.error("[SPARE_PARTS] ❌ No data source available - neither Google Drive nor local CSV found")
                    return False
                
                # Build cache
                self._build_cache(df)
                self.last_refresh = time.time()
                
                logger.info(f"[SPARE_PARTS] ✅ Cache refreshed: {self.part_count} parts loaded")
                return True
                
            except Exception as e:
                logger.error(f"[SPARE_PARTS] Refresh failed: {e}", exc_info=True)
                return False
            finally:
                self.is_refreshing = False
    
    def _build_cache(self, df: pd.DataFrame):
        """Build search indexes from DataFrame."""
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        # Find the part number column (flexible matching)
        part_col = None
        price_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if "part" in col_lower and ("no" in col_lower or "number" in col_lower or "#" in col):
                part_col = col
            elif "price" in col_lower or "dollar" in col_lower:
                price_col = col
        
        # Fallback to first two columns
        if not part_col:
            part_col = df.columns[0]
        if not price_col and len(df.columns) > 1:
            price_col = df.columns[1]
        
        logger.info(f"[SPARE_PARTS] Using columns: part='{part_col}', price='{price_col}'")
        
        # Clear existing data
        self.parts.clear()
        self.exact_index.clear()
        self.prefix_index.clear()
        self.base_model_index.clear()
        
        # Process each row
        for _, row in df.iterrows():
            raw_part = str(row.get(part_col, "")).strip()
            raw_price = str(row.get(price_col, "")).strip() if price_col else ""
            
            if not raw_part:
                continue
            
            # Normalize and parse
            normalized = normalize_part_number(raw_part)
            base_model = extract_base_model(normalized)
            price_numeric, has_price = parse_price(raw_price)
            
            # Create part info
            part_info = SparePartInfo(
                part_number=raw_part,
                normalized_number=normalized,
                price_raw=raw_price if raw_price else "$ -",
                price_numeric=price_numeric,
                has_price=has_price,
                is_obsolete=is_obsolete_part(raw_part),
                is_display_dummy=is_display_dummy(raw_part),
                base_model=base_model
            )
            
            # Store in primary dict
            self.parts[normalized] = part_info
            
            # Build exact index
            self.exact_index[normalized] = part_info
            
            # Build prefix index (first 8 chars for broader matching)
            prefix = normalized[:8] if len(normalized) >= 8 else normalized
            if prefix not in self.prefix_index:
                self.prefix_index[prefix] = []
            self.prefix_index[prefix].append(part_info)
            
            # Build base model index
            if base_model:
                if base_model not in self.base_model_index:
                    self.base_model_index[base_model] = []
                self.base_model_index[base_model].append(part_info)
        
        self.part_count = len(self.parts)
    
    def find_part(
        self,
        part_number: str,
        allow_fuzzy: bool = True,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Find spare part pricing with multiple search strategies.
        
        Returns dict with:
        - success: bool
        - parts: List of matching parts
        - search_method: How the match was found
        - message: Human-readable result
        """
        
        # Ensure cache is loaded
        if not self.parts:
            self.refresh()
        
        normalized = normalize_part_number(part_number)
        base_model = extract_base_model(normalized)
        
        # Strategy 1: Exact match
        if normalized in self.exact_index:
            part = self.exact_index[normalized]
            return self._format_result([part], "exact_match", f"Found exact match for {part_number}")
        
        # Strategy 2: Base model match (find all finish variants)
        if base_model in self.base_model_index:
            parts = self.base_model_index[base_model][:limit]
            return self._format_result(
                parts, 
                "base_model_match", 
                f"Found {len(parts)} finish variant(s) for {base_model}"
            )
        
        # Strategy 3: Prefix match
        prefix = normalized[:8] if len(normalized) >= 8 else normalized
        if prefix in self.prefix_index:
            parts = self.prefix_index[prefix][:limit]
            return self._format_result(
                parts,
                "prefix_match",
                f"Found {len(parts)} part(s) matching prefix '{prefix}'"
            )
        
        # Strategy 4: Fuzzy match
        if allow_fuzzy:
            fuzzy_matches = self._fuzzy_search(normalized, limit)
            if fuzzy_matches:
                return self._format_result(
                    fuzzy_matches,
                    "fuzzy_match",
                    f"Found {len(fuzzy_matches)} similar part(s) (approximate match)"
                )
        
        # No matches found
        return {
            "success": False,
            "parts": [],
            "search_method": "no_match",
            "message": f"No spare part found matching '{part_number}'. Please verify the part number.",
            "suggestions": self._get_suggestions(normalized)
        }
    
    def _fuzzy_search(self, query: str, limit: int) -> List[SparePartInfo]:
        """Find parts with similar names using fuzzy matching."""
        matches = []
        
        # Sample candidates for performance
        candidates = list(self.parts.keys())[:MAX_FUZZY_CANDIDATES]
        
        for normalized_key in candidates:
            similarity = SequenceMatcher(None, query, normalized_key).ratio()
            if similarity >= FUZZY_MATCH_THRESHOLD:
                matches.append((similarity, self.parts[normalized_key]))
        
        # Sort by similarity descending
        matches.sort(key=lambda x: x[0], reverse=True)
        
        return [m[1] for m in matches[:limit]]
    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get suggestions for mistyped part numbers."""
        # Find parts that start similarly
        prefix = query[:4] if len(query) >= 4 else query
        suggestions = []
        
        for normalized in self.parts.keys():
            if normalized.startswith(prefix):
                suggestions.append(self.parts[normalized].part_number)
                if len(suggestions) >= 5:
                    break
        
        return suggestions
    
    def _format_result(
        self, 
        parts: List[SparePartInfo], 
        method: str, 
        message: str
    ) -> Dict[str, Any]:
        """Format search result as dictionary."""
        
        formatted_parts = []
        for part in parts:
            formatted_parts.append({
                "part_number": part.part_number,
                "price": part.price_raw,
                "price_numeric": part.price_numeric,
                "has_price": part.has_price,
                "price_status": "available" if part.has_price else "not_set",
                "is_obsolete": part.is_obsolete,
                "is_display_dummy": part.is_display_dummy,
            })
        
        return {
            "success": True,
            "parts": formatted_parts,
            "count": len(formatted_parts),
            "search_method": method,
            "message": message
        }
    
    def start_background_refresh(self):
        """Start background thread for periodic refresh."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        
        def refresh_loop():
            logger.info("[SPARE_PARTS] Background refresh thread started")
            while True:
                try:
                    self.refresh()
                except Exception as e:
                    logger.error(f"[SPARE_PARTS] Background refresh error: {e}")
                
                # Sleep until next refresh
                time.sleep(REFRESH_INTERVAL_SECONDS)
        
        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

_cache: Optional[SparePartsPricingCache] = None


def get_spare_parts_cache() -> SparePartsPricingCache:
    """Get or create the spare parts pricing cache singleton."""
    global _cache
    if _cache is None:
        _cache = SparePartsPricingCache()
        _cache.refresh()
        _cache.start_background_refresh()
    return _cache


def find_spare_part_pricing(
    part_number: str,
    allow_fuzzy: bool = True,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Look up spare part pricing.
    
    Args:
        part_number: The spare part number to look up
        allow_fuzzy: Whether to allow fuzzy matching for typos
        limit: Maximum number of results to return
        
    Returns:
        Dict with success, parts list, and search metadata
    """
    cache = get_spare_parts_cache()
    return cache.find_part(part_number, allow_fuzzy=allow_fuzzy, limit=limit)


def get_all_parts_for_base_model(base_model: str) -> List[Dict[str, Any]]:
    """Get all finish variants for a base model number."""
    cache = get_spare_parts_cache()
    
    normalized = normalize_part_number(base_model)
    base = extract_base_model(normalized)
    
    if base in cache.base_model_index:
        parts = cache.base_model_index[base]
        return [
            {
                "part_number": p.part_number,
                "price": p.price_raw,
                "price_numeric": p.price_numeric,
                "has_price": p.has_price,
            }
            for p in parts
        ]
    
    return []


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_spare_parts_cache():
    """Initialize the cache on module load (called from main app startup)."""
    logger.info("[SPARE_PARTS] Initializing spare parts pricing cache...")
    cache = get_spare_parts_cache()
    logger.info(f"[SPARE_PARTS] Cache initialized with {cache.part_count} parts")
