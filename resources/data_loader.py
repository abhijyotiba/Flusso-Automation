"""
Data Loader Service - In-memory Product Database

Loads product catalog (CSV) and media assets (JSON) into memory
at startup for fast product lookup and retrieval.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fuzzywuzzy import fuzz


@dataclass
class ProductContext:
    """Structured product information context"""
    
    model_number: str
    specs: Dict[str, Any] = field(default_factory=dict)
    media: Dict[str, List[Dict]] = field(default_factory=dict)
    documents: List[Dict] = field(default_factory=list)
    matched_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "model_number": self.model_number,
            "specs": self.specs,
            "media": self.media,
            "documents": self.documents,
            "matched_confidence": self.matched_confidence
        }


class ProductDatabase:
    """
    In-memory product database with fast lookup capabilities.
    
    Loads data once at startup and provides fuzzy/regex matching
    for product model numbers in user queries.
    """
    
    def __init__(self, data_dir: str = "data"):
        # Resolve paths relative to THIS file's location
        # In Docker: /app/server/app/services/data_loader.py -> /app/server/
        # Locally: .../server/app/services/data_loader.py -> .../server/
        base_dir = Path(__file__).resolve().parent.parent.parent  # Goes up to /server/
        self.data_dir = base_dir / data_dir
        
        # Debug: print resolved path for troubleshooting
        print(f"📂 Data directory resolved to: {self.data_dir}")
        
        self.media_data: Dict[str, Any] = {}
        self.catalog_df: Optional[pd.DataFrame] = None
        self.model_index: Dict[str, str] = {}  # Normalized model -> Original model
        self.loaded = False
        
    def load_data(self) -> None:
        """Load JSON and Excel data into memory"""
        try:
            # Load media data (JSON) - metadata_manifest.json format
            media_path = self.data_dir / "metadata_manifest.json"
            if media_path.exists():
                with open(media_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    # Transform the data structure: extract metadata by Model_NO
                    for item in raw_data:
                        if 'metadata' in item and 'Model_NO' in item['metadata']:
                            model_no = item['metadata']['Model_NO']
                            # Store the complete item including originalUrl, savedAs, and metadata
                            self.media_data[model_no] = item
                print(f"✓ Loaded {len(self.media_data)} products from metadata_manifest.json")
            else:
                print(f"⚠ Media file not found: {media_path}")
            
            # Load catalog data (Excel) - Product-2025-11-12.xlsx
            catalog_path = self.data_dir / "Product-2025-11-12.xlsx"
            if catalog_path.exists():
                self.catalog_df = pd.read_excel(catalog_path)
                print(f"✓ Loaded {len(self.catalog_df)} products from Excel catalog")
            else:
                print(f"⚠ Catalog file not found: {catalog_path}")
            
            # Build model index for fast lookup
            self._build_model_index()
            
            self.loaded = True
            print(f"✓ Product database loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading product database: {e}")
            raise
    
    def _build_model_index(self) -> None:
        """Build normalized model number index for fuzzy matching"""
        # Index from CSV
        if self.catalog_df is not None:
            for model in self.catalog_df['Model_NO'].dropna().unique():
                normalized = self._normalize_model(model)
                self.model_index[normalized] = model
        
        # Index from JSON
        for model in self.media_data.keys():
            normalized = self._normalize_model(model)
            self.model_index[normalized] = model
        
        print(f"✓ Built model index with {len(self.model_index)} entries")
    
    @staticmethod
    def _normalize_model(model: str) -> str:
        """Normalize model number for matching (remove dots, spaces, lowercase)"""
        return re.sub(r'[.\s-]', '', model.lower())
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL to ensure it has proper protocol"""
        if not url or url == '':
            return ''
        # If URL doesn't start with http:// or https://, add https://
        if not url.startswith(('http://', 'https://')):
            return f'https://{url}'
        return url
    
    def find_product(self, query: str) -> Optional[ProductContext]:
        """
        Search for product model number in query using multiple strategies.
        
        Strategies:
        1. Exact match (normalized)
        2. Regex pattern matching (common model formats)
        3. Fuzzy matching on model numbers
        
        Args:
            query: User query string
            
        Returns:
            ProductContext if found, None otherwise
        """
        if not self.loaded:
            raise RuntimeError("Database not loaded. Call load_data() first.")
        
        query_normalized = query.lower()
        
        # Strategy 1: Check if any known model is in the query (exact substring)
        best_match = None
        best_confidence = 0.0
        
        for normalized_model, original_model in self.model_index.items():
            # Direct substring match
            if normalized_model in self._normalize_model(query):
                confidence = 1.0
                if confidence > best_confidence:
                    best_match = original_model
                    best_confidence = confidence
        
        # Strategy 2: Regex pattern matching for common model formats
        if not best_match:
            # Patterns like: GC-303-T, 10.FGC.4003CP, FF-1234-CP
            patterns = [
                r'\b([A-Z]{2,3}[-.]?\d{3,4}[-.]?[A-Z]{1,3})\b',  # GC-303-T, FF-1234-CP
                r'\b(\d{1,2}\.[A-Z]{2,3}\.\d{4}[A-Z]{2,3})\b',  # 10.FGC.4003CP
                r'\b([A-Z]{2}-\d{4}-[A-Z]{2,3})\b'              # SD-5678-BN
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    match_normalized = self._normalize_model(match)
                    if match_normalized in self.model_index:
                        best_match = self.model_index[match_normalized]
                        best_confidence = 0.95
                        break
                if best_match:
                    break
        
        # Strategy 3: Fuzzy matching as fallback
        if not best_match and len(query) > 5:
            for normalized_model, original_model in self.model_index.items():
                # Use fuzzy matching
                ratio = fuzz.partial_ratio(normalized_model, self._normalize_model(query))
                if ratio > 80 and ratio / 100.0 > best_confidence:
                    best_match = original_model
                    best_confidence = ratio / 100.0
        
        # If match found, build ProductContext
        if best_match and best_confidence > 0.5:
            return self._build_product_context(best_match, best_confidence)
        
        return None
    
    def _build_product_context(self, model_number: str, confidence: float) -> ProductContext:
        """Build complete ProductContext from all data sources"""
        
        # Get specs from Excel
        specs = {}
        if self.catalog_df is not None:
            product_row = self.catalog_df[self.catalog_df['Model_NO'] == model_number]
            if not product_row.empty:
                specs = product_row.iloc[0].to_dict()
                # Clean up NaN values
                specs = {k: v for k, v in specs.items() if pd.notna(v)}
        
        # Get media from JSON (metadata_manifest structure)
        media = {
            "videos": [],
            "images": []
        }
        documents = []

        if model_number in self.media_data:
            media_item = self.media_data[model_number]
            metadata = media_item.get('metadata', {})

            # Defensive: If media is not a dict, reset to default
            if not isinstance(media, dict):
                media = {"videos": [], "images": []}

            # Extract image information
            if 'savedAs' in media_item:
                media["images"].append({
                    "url": self._normalize_url(metadata.get('Image_URL', '')),
                    "filename": media_item['savedAs'],
                    "original_url": self._normalize_url(media_item.get('originalUrl', ''))
                })

            # Extract video links from metadata
            video_links = [
                metadata.get('Installation_video_Link', ''),
                metadata.get('Operational_Video_Link', ''),
                metadata.get('Lifestyle_Video_Link', '')
            ]
            media["videos"] = [self._normalize_url(v) for v in video_links if v and v != '']

            # Extract document information
            doc_fields = [
                ('Spec_Sheet_File_Name', 'Spec_Sheet_Full_URL', 'Specification Sheet'),
                ('Installation_Manual_File_Name', 'Installation_manual_Full_URL', 'Installation Manual'),
                ('Parts_Diagram_File_Name', 'Part_Diagram_Full_URL', 'Parts Diagram')
            ]

            for file_field, url_field, doc_type in doc_fields:
                filename = metadata.get(file_field)
                if filename and filename != '':
                    documents.append({
                        "type": doc_type,
                        "filename": filename,
                        "title": filename,  # Use filename as document name/title
                        "url": self._normalize_url(metadata.get(url_field, ''))
                    })

        # Defensive: Ensure media is a dict with required keys
        if not isinstance(media, dict):
            media = {"videos": [], "images": []}
        if "videos" not in media or not isinstance(media["videos"], list):
            media["videos"] = []
        if "images" not in media or not isinstance(media["images"], list):
            media["images"] = []

        return ProductContext(
            model_number=model_number,
            specs=specs,
            media=media,
            documents=documents,
            matched_confidence=confidence
        )
    
    def get_product_by_model(self, model_number: str) -> Optional[ProductContext]:
        """Get product by exact model number"""
        if model_number in self.model_index.values():
            return self._build_product_context(model_number, 1.0)
        return None
    
    def get_all_models(self) -> List[str]:
        """Return list of all known model numbers"""
        return list(set(self.model_index.values()))
    
    def search_by_category(self, category: str) -> List[Dict]:
        """Search products by category"""
        if self.catalog_df is None:
            return []
        
        matches = self.catalog_df[
            self.catalog_df['Product_Category'].str.contains(category, case=False, na=False) |
            self.catalog_df['Sub_Product_Category'].str.contains(category, case=False, na=False) |
            self.catalog_df['Sub_Sub_Product_Category'].str.contains(category, case=False, na=False)
        ]
        
        return matches.to_dict('records')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_products": len(self.model_index),
            "products_with_media": len(self.media_data),
            "products_with_specs": len(self.catalog_df) if self.catalog_df is not None else 0,
            "loaded": self.loaded
        }


# Global instance (initialized in main.py)
product_db: Optional[ProductDatabase] = None


def get_product_database() -> ProductDatabase:
    """Get global product database instance"""
    if product_db is None:
        raise RuntimeError("Product database not initialized")
    return product_db
