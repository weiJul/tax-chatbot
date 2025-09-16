"""
Text processing utilities for RAG Tax System
"""
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Utility class for text processing operations
    """
    
    @staticmethod
    def clean_text(text: str, remove_extra_whitespace: bool = True, 
                   normalize_unicode: bool = True) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Input text
            remove_extra_whitespace: Whether to remove extra whitespace
            normalize_unicode: Whether to normalize unicode characters
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode if requested
        if normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove excessive whitespace if requested
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    @staticmethod
    def extract_sections(text: str, section_patterns: List[str]) -> Dict[str, str]:
        """
        Extract sections from text based on patterns
        
        Args:
            text: Input text
            section_patterns: List of regex patterns for section headers
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            
            for i, match in enumerate(matches):
                section_name = match.group(1) if match.groups() else f"Section_{i+1}"
                start_pos = match.end()
                
                # Find end position (next section or end of text)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    # Look for next section from other patterns
                    next_section_pos = len(text)
                    for other_pattern in section_patterns:
                        if other_pattern != pattern:
                            other_matches = list(re.finditer(other_pattern, text[start_pos:], 
                                                           re.IGNORECASE | re.MULTILINE))
                            if other_matches:
                                next_section_pos = min(next_section_pos, start_pos + other_matches[0].start())
                    end_pos = next_section_pos
                
                section_content = text[start_pos:end_pos].strip()
                sections[section_name] = section_content
        
        return sections
    
    @staticmethod
    def extract_key_value_pairs(text: str, patterns: List[str]) -> Dict[str, str]:
        """
        Extract key-value pairs from text using patterns
        
        Args:
            text: Input text
            patterns: List of regex patterns for key-value extraction
            
        Returns:
            Dictionary of extracted key-value pairs
        """
        pairs = {}
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    pairs[key] = value
        
        return pairs
    
    @staticmethod
    def split_into_sentences(text: str, min_length: int = 10) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            min_length: Minimum sentence length
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= min_length:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    @staticmethod
    def extract_numbers(text: str, include_currency: bool = True) -> List[Dict[str, Any]]:
        """
        Extract numbers and numeric values from text
        
        Args:
            text: Input text
            include_currency: Whether to extract currency amounts
            
        Returns:
            List of dictionaries with number information
        """
        numbers = []
        
        # Currency patterns
        if include_currency:
            currency_patterns = [
                r'\$([0-9,]+\.?[0-9]*)',
                r'([0-9,]+\.?[0-9]*)\s*dollars?',
                r'([0-9,]+\.?[0-9]*)\s*USD'
            ]
            
            for pattern in currency_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value_str = match.group(1).replace(',', '')
                    try:
                        value = float(value_str)
                        numbers.append({
                            'type': 'currency',
                            'value': value,
                            'text': match.group(0),
                            'position': match.start()
                        })
                    except ValueError:
                        continue
        
        # Percentage patterns
        percentage_pattern = r'([0-9]+\.?[0-9]*)\s*%'
        matches = re.finditer(percentage_pattern, text)
        for match in matches:
            try:
                value = float(match.group(1))
                numbers.append({
                    'type': 'percentage',
                    'value': value,
                    'text': match.group(0),
                    'position': match.start()
                })
            except ValueError:
                continue
        
        # General numbers
        number_pattern = r'\b([0-9,]+\.?[0-9]*)\b'
        matches = re.finditer(number_pattern, text)
        for match in matches:
            value_str = match.group(1).replace(',', '')
            try:
                value = float(value_str)
                numbers.append({
                    'type': 'number',
                    'value': value,
                    'text': match.group(0),
                    'position': match.start()
                })
            except ValueError:
                continue
        
        # Sort by position and remove duplicates
        numbers.sort(key=lambda x: x['position'])
        unique_numbers = []
        seen_positions = set()
        
        for num in numbers:
            if num['position'] not in seen_positions:
                unique_numbers.append(num)
                seen_positions.add(num['position'])
        
        return unique_numbers
    
    @staticmethod
    def extract_dates(text: str) -> List[Dict[str, Any]]:
        """
        Extract dates from text
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with date information
        """
        dates = []
        
        date_patterns = [
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', 'MM/DD/YYYY'),
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', 'YYYY-MM-DD'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', 'Month DD, YYYY'),
            (r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', 'DD Month YYYY')
        ]
        
        for pattern, format_type in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append({
                    'text': match.group(0),
                    'format': format_type,
                    'position': match.start(),
                    'groups': match.groups()
                })
        
        return dates
    
    @staticmethod
    def extract_tax_terms(text: str) -> List[Dict[str, Any]]:
        """
        Extract tax-related terms and concepts
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries with tax term information
        """
        tax_terms = []
        
        # Define tax-related patterns
        tax_patterns = {
            'tax_types': r'\b(B&O|business and occupation|retail sales|use tax|property tax|excise tax|income tax|sales tax)\b',
            'tax_rates': r'\b([0-9]+\.?[0-9]*)\s*%\s*(tax|rate)\b',
            'filing_periods': r'\b(monthly|quarterly|annual|annually)\s*(filing|return|report)\b',
            'due_dates': r'\b(due|deadline|must be filed)\s*(by|before|on)\s*([A-Za-z]+ \d{1,2}|\d{1,2}/\d{1,2})\b',
            'exemptions': r'\b(exempt|exemption|deduction|credit)\s*(from|for)?\s*([a-zA-Z\s]+)\b',
            'penalties': r'\b(penalty|penalties|interest|late fee)\s*(of|for)?\s*([0-9%$\s,]+)\b'
        }
        
        for term_type, pattern in tax_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                tax_terms.append({
                    'type': term_type,
                    'text': match.group(0),
                    'position': match.start(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
        
        return tax_terms
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict[str, Any]:
        """
        Get basic text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'character_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'paragraph_count': 0,
                'average_words_per_sentence': 0,
                'average_chars_per_word': 0
            }
        
        # Basic counts
        character_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = TextProcessor.split_into_sentences(text)
        sentence_count = len(sentences)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Averages
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        avg_chars_per_word = character_count / word_count if word_count > 0 else 0
        
        return {
            'character_count': character_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'average_words_per_sentence': round(avg_words_per_sentence, 2),
            'average_chars_per_word': round(avg_chars_per_word, 2)
        }
    
    @staticmethod
    def highlight_terms(text: str, terms: List[str], 
                       highlight_start: str = "**", highlight_end: str = "**") -> str:
        """
        Highlight specific terms in text
        
        Args:
            text: Input text
            terms: List of terms to highlight
            highlight_start: Start marker for highlighting
            highlight_end: End marker for highlighting
            
        Returns:
            Text with highlighted terms
        """
        highlighted_text = text
        
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f"{highlight_start}{term}{highlight_end}", 
                highlighted_text
            )
        
        return highlighted_text


# Global text processor instance
text_processor = TextProcessor()