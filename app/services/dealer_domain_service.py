"""
Dealer Domain Service
Identifies if an email belongs to a dealer based on domain lookup.
Supports Google Drive sync with local CSV fallback (like spare_parts_pricing_service).
"""

import os
import io
import time
import logging
import threading
from typing import Optional, Set, Dict, Any
from functools import lru_cache

import pandas as pd

from app.config.settings import settings

logger = logging.getLogger(__name__)

# Configuration
DEALER_DOMAINS_SHEET_FILE_ID = getattr(settings, 'dealer_domains_sheet_file_id', None) or os.getenv("DEALER_DOMAINS_SHEET_FILE_ID")
DEALER_DOMAINS_REFRESH_HOURS = int(getattr(settings, 'dealer_domains_refresh_hours', 24) or os.getenv("DEALER_DOMAINS_REFRESH_HOURS", "24"))


def _get_drive_service():
    """
    Create Google Drive API service using Service Account credentials.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if creds_path and os.path.exists(creds_path):
            logger.info(f"[DEALER_DOMAINS] Using service account from: {creds_path}")
            credentials = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
        else:
            import google.auth
            logger.info("[DEALER_DOMAINS] Using default Google Cloud credentials")
            credentials, project = google.auth.default(
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
        
        service = build('drive', 'v3', credentials=credentials)
        return service
        
    except ImportError as e:
        logger.warning(f"[DEALER_DOMAINS] Google API libraries not installed: {e}")
        logger.info("[DEALER_DOMAINS] Will use local CSV fallback instead.")
        return None
    except Exception as e:
        error_msg = str(e)
        if "credentials were not found" in error_msg.lower() or "could not automatically determine" in error_msg.lower():
            logger.warning(
                f"[DEALER_DOMAINS] ⚠️ Google credentials not configured.\n"
                f"  → Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file path.\n"
                f"  → Will use local CSV fallback instead."
            )
        else:
            logger.warning(f"[DEALER_DOMAINS] ⚠️ Failed to initialize Google Drive service: {error_msg[:150]}")
            logger.info("[DEALER_DOMAINS] Will use local CSV fallback instead.")
        return None


def _download_sheet_from_drive(file_id: str) -> Optional[pd.DataFrame]:
    """
    Download dealer domains sheet from Google Drive.
    """
    if not file_id:
        logger.warning("[DEALER_DOMAINS] No DEALER_DOMAINS_SHEET_FILE_ID configured")
        return None
    
    service = _get_drive_service()
    if not service:
        return None
    
    try:
        from googleapiclient.http import MediaIoBaseDownload
        
        logger.info(f"[DEALER_DOMAINS] Downloading sheet from Google Drive (ID: {file_id[:10]}...)")
        
        request = service.files().export_media(
            fileId=file_id,
            mimeType='text/csv'
        )
        
        file_buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.debug(f"[DEALER_DOMAINS] Download progress: {int(status.progress() * 100)}%")
        
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer)
        logger.info(f"[DEALER_DOMAINS] Downloaded {len(df)} rows from Google Drive")
        
        return df
    
    except Exception as e:
        error_msg = str(e)
        
        if "HttpError 404" in error_msg or "File not found" in error_msg:
            logger.warning(
                f"[DEALER_DOMAINS] ⚠️ Google Drive file not accessible (404 Not Found).\n"
                f"  → File ID: {file_id}\n"
                f"  → This usually means:\n"
                f"    1. The file doesn't exist, OR\n"
                f"    2. The service account doesn't have access to this file.\n"
                f"  → Solution: Share the Google Sheet with the service account email address.\n"
                f"  → Falling back to local CSV file..."
            )
        elif "HttpError 403" in error_msg or "forbidden" in error_msg.lower():
            logger.warning(
                f"[DEALER_DOMAINS] ⚠️ Google Drive access forbidden (403).\n"
                f"  → The service account lacks permission to access this file.\n"
                f"  → Solution: Share the Google Sheet with the service account email (with Viewer access).\n"
                f"  → Falling back to local CSV file..."
            )
        else:
            logger.warning(f"[DEALER_DOMAINS] ⚠️ Google Drive download failed: {error_msg[:200]}")
            logger.warning(f"[DEALER_DOMAINS] Falling back to local CSV file...")
        
        return None


def _load_local_csv_fallback() -> Optional[pd.DataFrame]:
    """
    Fallback: Load dealer domains from local CSV file.
    """
    local_paths = [
        ("data/dealer_domains.csv", "csv"),
        ("data/Dealer-Email-Domains - Sheet1.csv", "csv"),
        ("data/Dealer-Email-Domains.csv", "csv"),
        ("Dealer-Email-Domains - Sheet1.csv", "csv"),
    ]
    
    for path, file_type in local_paths:
        if os.path.exists(path):
            try:
                logger.info(f"[DEALER_DOMAINS] Loading from local file: {path}")
                df = pd.read_csv(path)
                logger.info(f"[DEALER_DOMAINS] ✅ Loaded {len(df)} rows from local CSV")
                return df
            except Exception as e:
                logger.error(f"[DEALER_DOMAINS] Failed to read {path}: {e}")
                continue
    
    logger.error("[DEALER_DOMAINS] ❌ No local CSV file found")
    return None


def _extract_domain_from_email(email: str) -> str:
    """
    Extract the domain part from an email address.
    
    Examples:
        john@example.com → example.com
        050@hajoca.com → hajoca.com
        john.doe@sub.domain.com → sub.domain.com
    """
    if not email or "@" not in email:
        return ""
    
    # Get everything after @ and lowercase
    domain = email.lower().split("@")[-1].strip()
    return domain


def _normalize_dealer_entry(entry: str) -> str:
    """
    Normalize a dealer domain entry from the CSV.
    
    The CSV contains entries like:
        - "050@hajoca.com" (full email pattern)
        - "accounting@cabochontile.com" (full email)
        - "example.com" (just domain)
    
    We extract and normalize to just the domain part.
    """
    if not entry:
        return ""
    
    entry = str(entry).strip().lower()
    
    # If it contains @, extract domain part
    if "@" in entry:
        return entry.split("@")[-1]
    
    # Otherwise assume it's already a domain
    return entry


class DealerDomainCache:
    """
    In-memory cache for dealer domains with background refresh.
    """
    
    def __init__(self):
        self._domains: Set[str] = set()
        self._email_patterns: Set[str] = set()  # Full email patterns like "050@hajoca.com"
        self.last_refresh: Optional[float] = None
        self.is_refreshing: bool = False
        self._lock = threading.Lock()
        self._refresh_thread: Optional[threading.Thread] = None
    
    @property
    def domain_count(self) -> int:
        return len(self._domains)
    
    @property
    def pattern_count(self) -> int:
        return len(self._email_patterns)
    
    def _build_cache(self, df: pd.DataFrame) -> None:
        """Build domain and pattern sets from DataFrame."""
        domains = set()
        patterns = set()
        
        # Common public email domains - don't add these to domain match list
        # Only match on exact email pattern for these
        PUBLIC_EMAIL_DOMAINS = {
            "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
            "icloud.com", "me.com", "mac.com", "live.com", "msn.com",
            "comcast.net", "verizon.net", "att.net", "sbcglobal.net",
            "cox.net", "charter.net", "earthlink.net", "optonline.net",
            "mail.com", "protonmail.com", "zoho.com", "ymail.com"
        }
        
        # Try to find the column with domain data
        col_name = None
        for possible_name in ["Dealer Domains", "dealer_domains", "Domain", "domain", "Email", "email"]:
            if possible_name in df.columns:
                col_name = possible_name
                break
        
        if col_name is None and len(df.columns) > 0:
            col_name = df.columns[0]
            logger.info(f"[DEALER_DOMAINS] Using first column: '{col_name}'")
        
        if col_name is None:
            logger.error("[DEALER_DOMAINS] No suitable column found in CSV")
            return
        
        for entry in df[col_name].dropna():
            entry_str = str(entry).strip().lower()
            
            if not entry_str or entry_str == "nan":
                continue
            
            # Store the full pattern (for exact email matching)
            if "@" in entry_str:
                patterns.add(entry_str)
                # Only add domain if it's NOT a public email provider
                domain = entry_str.split("@")[-1]
                if domain and domain not in PUBLIC_EMAIL_DOMAINS:
                    domains.add(domain)
            else:
                # It's just a domain - add directly
                domains.add(entry_str)
        
        self._domains = domains
        self._email_patterns = patterns
        
        # Log public domains excluded
        public_count = len([p for p in patterns if any(pub in p for pub in PUBLIC_EMAIL_DOMAINS)])
        logger.info(f"[DEALER_DOMAINS] Cache built: {len(domains)} unique domains, {len(patterns)} email patterns")
        logger.info(f"[DEALER_DOMAINS] Note: {public_count} entries use public email domains (gmail, etc.) - exact match only")
        
        logger.info(f"[DEALER_DOMAINS] Cache built: {len(domains)} unique domains, {len(patterns)} email patterns")
    
    def is_dealer_email(self, email: str) -> bool:
        """
        Check if an email belongs to a dealer.
        
        Matching strategy:
        1. First check exact email pattern match (e.g., "050@hajoca.com")
        2. Then check domain match (e.g., "hajoca.com")
        """
        if not email:
            return False
        
        email_lower = email.lower().strip()
        
        # Ensure cache is loaded
        if not self._domains and not self._email_patterns:
            self.refresh()
        
        # Check exact email pattern first
        if email_lower in self._email_patterns:
            return True
        
        # Check domain
        domain = _extract_domain_from_email(email_lower)
        if domain and domain in self._domains:
            return True
        
        return False
    
    def get_matching_info(self, email: str) -> Dict[str, Any]:
        """
        Get detailed matching information for an email.
        """
        if not email:
            return {"is_dealer": False, "match_type": None, "matched_value": None}
        
        email_lower = email.lower().strip()
        
        # Ensure cache is loaded
        if not self._domains and not self._email_patterns:
            self.refresh()
        
        # Check exact email pattern first
        if email_lower in self._email_patterns:
            return {
                "is_dealer": True,
                "match_type": "exact_email_pattern",
                "matched_value": email_lower
            }
        
        # Check domain
        domain = _extract_domain_from_email(email_lower)
        if domain and domain in self._domains:
            return {
                "is_dealer": True,
                "match_type": "domain_match",
                "matched_value": domain
            }
        
        return {
            "is_dealer": False,
            "match_type": None,
            "matched_value": None,
            "email_domain": domain
        }
    
    def refresh(self, force: bool = False) -> bool:
        """
        Refresh the dealer domains cache.
        """
        if self.is_refreshing:
            logger.debug("[DEALER_DOMAINS] Refresh already in progress")
            return True
        
        # Check if refresh needed
        if not force and self.last_refresh:
            age_hours = (time.time() - self.last_refresh) / 3600
            if age_hours < DEALER_DOMAINS_REFRESH_HOURS:
                logger.debug(f"[DEALER_DOMAINS] Cache still fresh ({age_hours:.1f}h old)")
                return True
        
        with self._lock:
            self.is_refreshing = True
            
            try:
                # Try Google Drive first
                df = _download_sheet_from_drive(DEALER_DOMAINS_SHEET_FILE_ID)
                
                # Fallback to local CSV
                if df is None:
                    df = _load_local_csv_fallback()
                
                if df is None:
                    logger.error("[DEALER_DOMAINS] ❌ No data source available")
                    return False
                
                # Build cache
                self._build_cache(df)
                self.last_refresh = time.time()
                
                logger.info(f"[DEALER_DOMAINS] ✅ Cache refreshed: {self.domain_count} domains loaded")
                return True
                
            except Exception as e:
                logger.error(f"[DEALER_DOMAINS] Refresh failed: {e}", exc_info=True)
                return False
            finally:
                self.is_refreshing = False
    
    def start_background_refresh(self) -> None:
        """Start background refresh thread."""
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        
        def refresh_loop():
            while True:
                time.sleep(DEALER_DOMAINS_REFRESH_HOURS * 3600)
                try:
                    self.refresh(force=True)
                except Exception as e:
                    logger.error(f"[DEALER_DOMAINS] Background refresh error: {e}")
        
        self._refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        self._refresh_thread.start()
        logger.info("[DEALER_DOMAINS] Background refresh thread started")


# Global cache instance
_dealer_domain_cache: Optional[DealerDomainCache] = None


def get_dealer_domain_cache() -> DealerDomainCache:
    """Get or create the global dealer domain cache."""
    global _dealer_domain_cache
    
    if _dealer_domain_cache is None:
        _dealer_domain_cache = DealerDomainCache()
        _dealer_domain_cache.refresh()
        _dealer_domain_cache.start_background_refresh()
    
    return _dealer_domain_cache


def is_dealer_email(email: str) -> bool:
    """
    Check if an email belongs to a dealer.
    
    Args:
        email: The email address to check
        
    Returns:
        True if the email domain is in the dealer list
    """
    cache = get_dealer_domain_cache()
    return cache.is_dealer_email(email)


def get_dealer_match_info(email: str) -> Dict[str, Any]:
    """
    Get detailed information about dealer email matching.
    
    Args:
        email: The email address to check
        
    Returns:
        Dict with is_dealer, match_type, matched_value
    """
    cache = get_dealer_domain_cache()
    return cache.get_matching_info(email)


def get_dealer_domain_stats() -> Dict[str, Any]:
    """
    Get statistics about the dealer domain cache.
    """
    cache = get_dealer_domain_cache()
    return {
        "domain_count": cache.domain_count,
        "pattern_count": cache.pattern_count,
        "last_refresh": cache.last_refresh,
        "is_refreshing": cache.is_refreshing
    }
