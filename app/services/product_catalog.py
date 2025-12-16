"""
Product Catalog Service - Comprehensive In-Memory Product Database
Loads product data from JSON manifest and provides fast multi-strategy search.

Features:
- Multiple search indexes (model, group, category, collection, finish, keywords)
- Exact, prefix, fuzzy, and keyword-based search
- Finish code to name mapping
- Group/variation awareness
- Rich product data with 70 fields

Data Source: metadata_manifest.json (5,687 products)
"""

import json
import logging
import os
import re
import threading
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to the JSON manifest file (relative to project root)
JSON_MANIFEST_PATH = "data/metadata_manifest.json"

# Search configuration
FUZZY_MATCH_THRESHOLD = 0.75  # Minimum similarity for fuzzy matches
MAX_FUZZY_CANDIDATES = 100     # Max candidates to check for fuzzy match
MAX_RESULTS_DEFAULT = 10       # Default number of results to return

# =============================================================================
# FINISH CODE MAPPING
# =============================================================================
FINISH_CODE_MAP = {
    "CP": "Chrome",
    "BN": "Brushed Nickel PVD",
    "PN": "Polished Nickel PVD",
    "MB": "Matte Black",
    "SB": "Satin Brass PVD",
    "BB": "Brushed Bronze PVD",
    "GW": "Gloss White",
    "GB": "Gloss Black",
    "SS": "Stainless Steel",
    "RB": "Rough Brass",
    "BG": "Brushed Gold PVD",
    "PS": "Polished Steel",
    "AG": "Army Green",
    "BP": "Blue Platinum",
    "CR": "Crimson",
    "DG": "Dark Grey",
    "DGR": "Dark Green",
    "DR": "Deep Red",
    "DT": "Dark Tan",
    "GMG": "Gun Metal Grey",
    "IG": "Isenberg Green",
    "LG": "Leaf Green",
    "LT": "Light Tan",
    "LV": "Light Verde",
    "NB": "Navy Blue",
    "RG": "Rock Grey",
    "SG": "Steel Grey",
    "SKB": "Sky Blue",
    "VB": "Vortex Brown",
}

# Reverse mapping for finish name to code
FINISH_NAME_TO_CODE = {v.upper(): k for k, v in FINISH_CODE_MAP.items()}


# =============================================================================
# GLOBAL STATE
# =============================================================================
class ProductCatalog:
    """In-memory product catalog with multiple search indexes."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single catalog instance."""
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
        self.products: List[Dict[str, Any]] = []
        
        # Primary indexes
        self.model_index: Dict[str, Dict[str, Any]] = {}  # MODEL_NO -> product
        self.group_index: Dict[str, List[Dict[str, Any]]] = {}  # GROUP -> [products]
        
        # Secondary indexes
        self.category_index: Dict[str, List[Dict[str, Any]]] = {}
        self.sub_category_index: Dict[str, List[Dict[str, Any]]] = {}
        self.collection_index: Dict[str, List[Dict[str, Any]]] = {}
        self.finish_index: Dict[str, List[Dict[str, Any]]] = {}
        
        # Keyword index (inverted index for text search)
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> set of MODEL_NOs
        
        # Model numbers list for fuzzy search
        self.all_model_numbers: List[str] = []
        
        # Statistics
        self.stats = {
            "total_products": 0,
            "total_groups": 0,
            "total_categories": 0,
            "total_collections": 0,
            "load_time_ms": 0,
            "last_loaded": None
        }
        
        logger.info("[PRODUCT_CATALOG] ProductCatalog instance created")
    
    def load_from_json(self, json_path: Optional[str] = None) -> bool:
        """
        Load product data from JSON manifest file.
        
        Args:
            json_path: Path to JSON file. If None, uses default path.
            
        Returns:
            True if loaded successfully, False otherwise.
        """
        start_time = time.time()
        
        if json_path is None:
            # Try to find the JSON file relative to project root
            project_root = Path(__file__).parent.parent.parent
            json_path = project_root / JSON_MANIFEST_PATH
        else:
            json_path = Path(json_path)
        
        if not json_path.exists():
            logger.error(f"[PRODUCT_CATALOG] JSON file not found: {json_path}")
            return False
        
        try:
            logger.info(f"[PRODUCT_CATALOG] Loading products from: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            logger.info(f"[PRODUCT_CATALOG] Loaded {len(raw_data)} raw records")
            
            # Parse and index products
            self._clear_indexes()
            
            for item in raw_data:
                metadata = item.get("metadata", {})
                if not metadata:
                    continue
                
                # Normalize and store product
                product = self._normalize_product(metadata, item)
                if product and product.get("model_no"):
                    self.products.append(product)
                    self._index_product(product)
            
            # Build keyword index
            self._build_keyword_index()
            
            # Update statistics
            load_time = (time.time() - start_time) * 1000
            self.stats = {
                "total_products": len(self.products),
                "total_groups": len(self.group_index),
                "total_categories": len(self.category_index),
                "total_collections": len(self.collection_index),
                "load_time_ms": round(load_time, 2),
                "last_loaded": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"[PRODUCT_CATALOG] ✅ Loaded {self.stats['total_products']} products "
                       f"in {load_time:.0f}ms ({self.stats['total_groups']} groups)")
            
            return True
            
        except Exception as e:
            logger.error(f"[PRODUCT_CATALOG] Failed to load JSON: {e}", exc_info=True)
            return False
    
    def _clear_indexes(self):
        """Clear all indexes."""
        self.products = []
        self.model_index = {}
        self.group_index = {}
        self.category_index = {}
        self.sub_category_index = {}
        self.collection_index = {}
        self.finish_index = {}
        self.keyword_index = {}
        self.all_model_numbers = []
    
    def _normalize_product(self, metadata: Dict[str, Any], raw_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize product data into a clean, consistent structure.
        
        Extracts and normalizes all 70 fields into a usable format.
        """
        model_no = str(metadata.get("Model_NO", "")).strip().upper()
        if not model_no:
            return None
        
        group_no = str(metadata.get("Common_Group_Number", "")).strip().upper()
        
        # Extract finish code from model number
        finish_code = ""
        finish_name = metadata.get("Finish", "")
        if group_no and model_no.startswith(group_no):
            finish_code = model_no[len(group_no):].strip()
        
        # Build feature bullets list
        features = []
        for i in range(1, 7):
            bullet = metadata.get(f"Description Bullet {i}", "")
            if bullet and str(bullet).strip():
                features.append(str(bullet).strip())
        
        # Parse dimensions
        def safe_float(val, default=0.0):
            try:
                return float(val) if val else default
            except (ValueError, TypeError):
                return default
        
        # Build normalized product
        product = {
            # Core identification
            "model_no": model_no,
            "group_number": group_no or model_no,
            "main_model_number": str(metadata.get("Main_Model_Number", "")).strip(),
            "upc": str(metadata.get("Item_UPC_Number", "")).strip(),
            
            # Product info
            "title": str(metadata.get("Product_Title", "")).strip(),
            "description": str(metadata.get("Description", "")).strip(),
            "keywords": str(metadata.get("Keywords", "")).strip(),
            "features": features,
            
            # Classification
            "category": str(metadata.get("Product_Category", "")).strip(),
            "sub_category": str(metadata.get("Sub_Product_Category", "")).strip(),
            "sub_sub_category": str(metadata.get("Sub_Sub_Product_Category", "")).strip(),
            "collection": str(metadata.get("Collection", "")).strip(),
            "style": str(metadata.get("Style", "")).strip(),
            
            # Finish
            "finish_code": finish_code,
            "finish_name": finish_name,
            
            # Pricing
            "list_price": safe_float(metadata.get("List_Price")),
            "map_price": safe_float(metadata.get("MAP_Price")),
            "cad_price": safe_float(metadata.get("CAD_List_Price")),
            
            # Specifications
            "flow_rate_gpm": safe_float(metadata.get("Flow_Rate_GPM")),
            "holes_needed": int(safe_float(metadata.get("Holes_Needed_For_Installation"))),
            "height_inches": safe_float(metadata.get("Product_Height_Inches")),
            "length_inches": safe_float(metadata.get("Product_Length_Inches")),
            "width_inches": safe_float(metadata.get("Product_Width_Inches")),
            "weight_lbs": safe_float(metadata.get("Package_Weight_lbs")),
            "is_touch_capable": str(metadata.get("IS_Touch_Capable", "")).upper() == "TRUE",
            
            # Status flags
            "status": str(metadata.get("Product_Status", "")).strip(),
            "is_active": str(metadata.get("Product_Status", "")).upper() in ["ACTIVE", ""],
            "is_spare_part": str(metadata.get("Is_Spare_Part", "")).upper() == "TRUE",
            "is_special_finish": str(metadata.get("Is_Special_Finish", "")).upper() == "TRUE",
            "display_on_website": str(metadata.get("Display_On_Website", "")).upper() == "YES",
            "can_sell_online": str(metadata.get("Can_Sell_Online", "")).upper() == "YES",
            
            # URLs
            "product_url": self._ensure_https(metadata.get("product_url", "")),
            "image_url": self._ensure_https(metadata.get("Image_URL", "")),
            "collection_url": str(metadata.get("Collection_URL", "")).strip(),
            
            # Document URLs
            "spec_sheet_url": self._ensure_https(metadata.get("Spec_Sheet_Full_URL", "")),
            "install_manual_url": self._ensure_https(metadata.get("Installation_manual_Full_URL", "")),
            "parts_diagram_url": self._ensure_https(metadata.get("Part_Diagram_Full_URL", "")),
            
            # Document filenames
            "spec_sheet_file": str(metadata.get("Spec_Sheet_File_Name", "")).strip(),
            "install_manual_file": str(metadata.get("Installation_Manual_File_Name", "")).strip(),
            "parts_diagram_file": str(metadata.get("Parts_Diagram_File_Name", "")).strip(),
            
            # Video URLs
            "install_video_url": str(metadata.get("Installation_video_Link", "")).strip(),
            "operational_video_url": str(metadata.get("Operational_Video_Link", "")).strip(),
            "lifestyle_video_url": str(metadata.get("Lifestyle_Video_Link", "")).strip(),
            
            # Warranty
            "warranty": str(metadata.get("Warranty", "")).strip(),
            
            # Popularity (for ranking)
            "popularity": int(safe_float(metadata.get("Popularity", 0))),
        }
        
        return product
    
    def _ensure_https(self, url: str) -> str:
        """Ensure URL has https:// prefix."""
        if not url:
            return ""
        url = str(url).strip()
        if url and not url.startswith("http"):
            return f"https://{url}"
        return url
    
    def _index_product(self, product: Dict[str, Any]):
        """Add product to all relevant indexes."""
        model_no = product["model_no"]
        
        # Model index (primary)
        self.model_index[model_no] = product
        self.all_model_numbers.append(model_no)
        
        # Group index
        group = product["group_number"]
        if group not in self.group_index:
            self.group_index[group] = []
        self.group_index[group].append(product)
        
        # Category index
        category = product["category"].upper()
        if category:
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(product)
        
        # Sub-category index
        sub_cat = product["sub_category"].upper()
        if sub_cat:
            if sub_cat not in self.sub_category_index:
                self.sub_category_index[sub_cat] = []
            self.sub_category_index[sub_cat].append(product)
        
        # Collection index
        collection = product["collection"].upper()
        if collection:
            if collection not in self.collection_index:
                self.collection_index[collection] = []
            self.collection_index[collection].append(product)
        
        # Finish index
        finish = product["finish_code"].upper()
        if finish:
            if finish not in self.finish_index:
                self.finish_index[finish] = []
            self.finish_index[finish].append(product)
    
    def _build_keyword_index(self):
        """Build inverted index for keyword search."""
        logger.info("[PRODUCT_CATALOG] Building keyword index...")
        
        for product in self.products:
            model_no = product["model_no"]
            
            # Combine searchable text
            searchable = " ".join([
                product["model_no"],
                product["group_number"],
                product["title"],
                product["keywords"],
                product["category"],
                product["sub_category"],
                product["collection"],
                product["finish_name"],
                " ".join(product["features"]),
            ]).lower()
            
            # Tokenize and index
            tokens = re.findall(r'\b[a-z0-9]+(?:\.[a-z0-9]+)*\b', searchable)
            for token in tokens:
                if len(token) >= 2:  # Skip single chars
                    if token not in self.keyword_index:
                        self.keyword_index[token] = set()
                    self.keyword_index[token].add(model_no)
        
        logger.info(f"[PRODUCT_CATALOG] Keyword index built with {len(self.keyword_index)} unique tokens")
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search_exact_model(self, model_no: str) -> Optional[Dict[str, Any]]:
        """
        Find product by exact model number match.
        
        Args:
            model_no: Model number to search for
            
        Returns:
            Product dict if found, None otherwise
        """
        normalized = model_no.strip().upper()
        return self.model_index.get(normalized)
    
    def search_by_group(self, group_no: str) -> List[Dict[str, Any]]:
        """
        Find all products in a group (all finish variations).
        
        Args:
            group_no: Group/base model number
            
        Returns:
            List of products in the group
        """
        normalized = group_no.strip().upper()
        
        # Direct group match
        if normalized in self.group_index:
            return self.group_index[normalized]
        
        # Try prefix match (group might be partial)
        matches = []
        for group_key, products in self.group_index.items():
            if group_key.startswith(normalized):
                matches.extend(products)
        
        return matches
    
    def search_prefix(self, prefix: str, limit: int = MAX_RESULTS_DEFAULT) -> List[Dict[str, Any]]:
        """
        Find products whose model number starts with the given prefix.
        
        Args:
            prefix: Model number prefix
            limit: Maximum results to return
            
        Returns:
            List of matching products
        """
        normalized = prefix.strip().upper()
        matches = []
        
        for model_no, product in self.model_index.items():
            if model_no.startswith(normalized):
                matches.append(product)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def search_fuzzy(self, query: str, threshold: float = FUZZY_MATCH_THRESHOLD, 
                     limit: int = MAX_RESULTS_DEFAULT) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find products with similar model numbers (handles typos).
        
        Args:
            query: Model number to search for (possibly with typos)
            threshold: Minimum similarity ratio (0.0-1.0)
            limit: Maximum results to return
            
        Returns:
            List of (product, similarity_score) tuples
        """
        normalized = query.strip().upper()
        candidates = []
        
        # First try prefix candidates (faster)
        prefix = normalized[:3] if len(normalized) >= 3 else normalized
        relevant_models = [m for m in self.all_model_numbers if m.startswith(prefix)]
        
        # If not enough prefix matches, check more
        if len(relevant_models) < MAX_FUZZY_CANDIDATES:
            relevant_models = self.all_model_numbers[:MAX_FUZZY_CANDIDATES * 2]
        
        for model_no in relevant_models[:MAX_FUZZY_CANDIDATES]:
            ratio = SequenceMatcher(None, normalized, model_no).ratio()
            if ratio >= threshold:
                candidates.append((self.model_index[model_no], ratio))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]
    
    def search_keywords(self, query: str, category: Optional[str] = None,
                       collection: Optional[str] = None,
                       limit: int = MAX_RESULTS_DEFAULT) -> List[Dict[str, Any]]:
        """
        Search products by keywords in title, description, features.
        
        Args:
            query: Search query string
            category: Optional category filter
            collection: Optional collection filter
            limit: Maximum results to return
            
        Returns:
            List of matching products (ranked by relevance)
        """
        # Tokenize query
        query_tokens = set(re.findall(r'\b[a-z0-9]+(?:\.[a-z0-9]+)*\b', query.lower()))
        
        if not query_tokens:
            return []
        
        # Find matching model numbers
        matching_models: Dict[str, int] = {}  # model_no -> match count
        
        for token in query_tokens:
            if token in self.keyword_index:
                for model_no in self.keyword_index[token]:
                    matching_models[model_no] = matching_models.get(model_no, 0) + 1
        
        if not matching_models:
            return []
        
        # Get products and apply filters
        results = []
        for model_no, score in sorted(matching_models.items(), key=lambda x: -x[1]):
            product = self.model_index.get(model_no)
            if not product:
                continue
            
            # Apply category filter
            if category and product["category"].upper() != category.upper():
                continue
            
            # Apply collection filter
            if collection and product["collection"].upper() != collection.upper():
                continue
            
            results.append(product)
            if len(results) >= limit:
                break
        
        return results
    
    def search_by_category(self, category: str, limit: int = MAX_RESULTS_DEFAULT) -> List[Dict[str, Any]]:
        """Get products by category."""
        normalized = category.strip().upper()
        products = self.category_index.get(normalized, [])
        return products[:limit]
    
    def search_by_collection(self, collection: str, limit: int = MAX_RESULTS_DEFAULT) -> List[Dict[str, Any]]:
        """Get products by collection."""
        normalized = collection.strip().upper()
        products = self.collection_index.get(normalized, [])
        return products[:limit]
    
    def get_finish_variations(self, group_no: str) -> Dict[str, str]:
        """
        Get all available finish variations for a product group.
        
        Args:
            group_no: Group/base model number
            
        Returns:
            Dict mapping finish code to finish name
        """
        products = self.search_by_group(group_no)
        variations = {}
        
        for product in products:
            code = product["finish_code"]
            name = product["finish_name"]
            if code:
                variations[code] = name
        
        return variations
    
    def get_related_parts(self, model_no: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find related spare parts for a product.
        
        Args:
            model_no: Product model number
            limit: Maximum parts to return
            
        Returns:
            List of related spare parts
        """
        normalized = model_no.strip().upper()
        product = self.model_index.get(normalized)
        
        if not product:
            return []
        
        group = product["group_number"]
        related = []
        
        # Search for parts with similar group number
        for part_model, part_product in self.model_index.items():
            if part_product["is_spare_part"]:
                if group in part_model or part_model.startswith(group[:7]):
                    related.append(part_product)
                    if len(related) >= limit:
                        break
        
        return related
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return sorted(self.category_index.keys())
    
    def get_collections(self) -> List[str]:
        """Get all available collections."""
        return sorted(self.collection_index.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return self.stats.copy()


# =============================================================================
# SINGLETON INSTANCE AND PUBLIC API
# =============================================================================

_catalog: Optional[ProductCatalog] = None


def get_product_catalog() -> ProductCatalog:
    """Get or create the singleton ProductCatalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = ProductCatalog()
    return _catalog


def init_product_catalog(json_path: Optional[str] = None) -> bool:
    """
    Initialize the product catalog.
    
    Should be called at application startup.
    
    Args:
        json_path: Optional path to JSON file
        
    Returns:
        True if successful
    """
    catalog = get_product_catalog()
    return catalog.load_from_json(json_path)


def ensure_catalog_loaded() -> ProductCatalog:
    """
    Ensure catalog is loaded, loading if necessary.
    
    Returns:
        The loaded ProductCatalog instance
    """
    catalog = get_product_catalog()
    if not catalog.products:
        catalog.load_from_json()
    return catalog


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def looks_like_model_number(text: str) -> bool:
    """
    Heuristic to detect if a string looks like a model/part number.
    
    Examples that match:
    - "100.1170", "100.1170CP", "10.FGC.4003BN"
    - "DKM.2420", "HS6270MB", "B.1200SS"
    
    Examples that don't match:
    - "floor mount faucet", "blue chrome finish"
    """
    if not text:
        return False
    
    s = text.strip()
    
    # Too long to be a model number
    if len(s) > 30:
        return False
    
    # Too many words (model numbers don't have spaces usually)
    if len(s.split()) > 2:
        return False
    
    # Allow letters, digits, dot, dash, slash
    if not all(ch.isalnum() or ch in ".-/" for ch in s.replace(" ", "")):
        return False
    
    # Must have at least one digit
    if not any(ch.isdigit() for ch in s):
        return False
    
    # Check for common model patterns
    patterns = [
        r'^\d{2,3}\.\d{4}',      # 100.1170
        r'^[A-Z]{1,3}\.\d{4}',   # B.1200
        r'^\d{2}\.[A-Z]{2,4}\.',  # 10.FGC.
        r'^[A-Z]{2,6}\d{3,5}',   # HS6270, DKM2420
    ]
    
    for pattern in patterns:
        if re.match(pattern, s.upper()):
            return True
    
    # If it has both letters and numbers with a dot or dash, likely a model
    has_letter = any(c.isalpha() for c in s)
    has_digit = any(c.isdigit() for c in s)
    has_separator = '.' in s or '-' in s
    
    return has_letter and has_digit and has_separator


def get_finish_name(code: str) -> str:
    """Get finish name from code."""
    return FINISH_CODE_MAP.get(code.upper(), code)


def get_finish_code(name: str) -> str:
    """Get finish code from name."""
    return FINISH_NAME_TO_CODE.get(name.upper(), "")
