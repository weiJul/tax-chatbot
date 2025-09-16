"""
Jurisdiction Detection Utility
Simple keyword-based detection for tax query jurisdiction classification
"""
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JurisdictionDetector:
    """
    Detects jurisdiction (California vs general) from user queries
    Uses simple keyword matching for reliable detection
    """
    
    def __init__(self):
        """Initialize jurisdiction detector with keyword patterns"""
        
        # California-specific keywords and patterns
        self.california_keywords = {
            # State identifiers
            "california", "ca", "calif", "golden state",
            
            # CA-specific tax terms
            "california tax", "ca tax", "california income tax", "ca income tax",
            "california state tax", "ca state tax", "california sales tax", "ca sales tax",
            "california withholding", "ca withholding",
            
            # CA tax forms
            "form 540", "540 form", "540ez", "540nr", "form 541", "541 form",
            "form 100", "100 form", "form 568", "568 form",
            
            # CA-specific agencies and programs
            "ftb", "franchise tax board", "california franchise tax board",
            "edd", "employment development department", "california edd",
            "cdtfa", "california department of tax and fee administration",
            "california disability insurance", "ca sdi", "sdi",
            
            # CA-specific tax concepts
            "california standard deduction", "ca standard deduction",
            "california exemptions", "ca exemptions",
            "california tax brackets", "ca tax brackets",
            "california tax rates", "ca tax rates",
            
            # References to pub29
            "pub29", "publication 29", "pub 29"
        }
        
        # Compile patterns for efficient matching
        self.california_pattern = self._compile_keyword_pattern(self.california_keywords)
        
        logger.info(f"Jurisdiction detector initialized with {len(self.california_keywords)} California keywords")
    
    def detect_jurisdiction(self, query: str) -> Optional[str]:
        """
        Detect jurisdiction from query text
        
        Args:
            query: User query text
            
        Returns:
            "california" if CA-specific terms detected, None for general
        """
        if not query or not query.strip():
            return None
        
        query_lower = query.lower()
        
        # Check for California patterns
        if self.california_pattern.search(query_lower):
            logger.debug(f"California jurisdiction detected in query: '{query[:50]}...'")
            return "california"
        
        # No jurisdiction detected - default to general
        logger.debug(f"No specific jurisdiction detected, using general: '{query[:50]}...'")
        return None
    
    def get_jurisdiction_confidence(self, query: str) -> Dict[str, float]:
        """
        Get confidence scores for different jurisdictions
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with jurisdiction confidence scores
        """
        if not query or not query.strip():
            return {"general": 1.0, "california": 0.0}
        
        query_lower = query.lower()
        
        # Count California keyword matches
        ca_matches = len(self.california_pattern.findall(query_lower))
        
        # Simple confidence scoring
        if ca_matches > 0:
            ca_confidence = min(0.9, 0.3 + (ca_matches * 0.2))  # Max 0.9 confidence
            general_confidence = 1.0 - ca_confidence
        else:
            ca_confidence = 0.0
            general_confidence = 1.0
        
        return {
            "general": general_confidence,
            "california": ca_confidence
        }
    
    def extract_jurisdiction_terms(self, query: str) -> List[str]:
        """
        Extract jurisdiction-specific terms found in the query
        
        Args:
            query: User query text
            
        Returns:
            List of jurisdiction-specific terms found
        """
        if not query or not query.strip():
            return []
        
        query_lower = query.lower()
        found_terms = []
        
        # Find all California terms
        for keyword in self.california_keywords:
            if keyword in query_lower:
                found_terms.append(keyword)
        
        return found_terms
    
    def _compile_keyword_pattern(self, keywords: set) -> re.Pattern:
        """
        Compile keywords into efficient regex pattern
        
        Args:
            keywords: Set of keywords to match
            
        Returns:
            Compiled regex pattern
        """
        # Sort by length (longest first) to match more specific terms first
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        # Escape special regex characters and create word boundary pattern
        escaped_keywords = []
        for keyword in sorted_keywords:
            escaped = re.escape(keyword)
            # Add word boundaries for better matching
            escaped_keywords.append(f"\\b{escaped}\\b")
        
        # Create alternation pattern
        pattern = "|".join(escaped_keywords)
        
        return re.compile(pattern, re.IGNORECASE)
    
    def add_jurisdiction_keywords(self, jurisdiction: str, keywords: List[str]) -> None:
        """
        Add new keywords for a jurisdiction (for future expansion)
        
        Args:
            jurisdiction: Jurisdiction name (e.g., "california", "texas")
            keywords: List of keywords to add
        """
        if jurisdiction.lower() == "california":
            self.california_keywords.update(k.lower() for k in keywords)
            # Recompile pattern
            self.california_pattern = self._compile_keyword_pattern(self.california_keywords)
            logger.info(f"Added {len(keywords)} keywords for California jurisdiction")
        else:
            logger.warning(f"Jurisdiction '{jurisdiction}' not currently supported")


# Global jurisdiction detector instance
jurisdiction_detector = JurisdictionDetector()