"""
Document processing module for RAG Tax System
Handles PDF processing, text extraction, and chunking
"""
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pypdf
except ImportError:
    pypdf = None

from ..utils.jurisdiction_detector import jurisdiction_detector
from ..utils.text_utils import text_processor
from .config import config_manager

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None


class DocumentProcessor:
    """
    Processes documents for RAG system
    Handles PDF extraction, cleaning, and chunking
    """
    
    def __init__(self):
        """Initialize document processor"""
        self.config = config_manager.document_processing
        
        # Create processed documents directory
        Path(self.config.processed_docs_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DocumentProcessor")
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if pypdf is None:
            raise ImportError("pypdf is required for PDF processing. Install it with: pip install pypdf")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            text_content = ""
            metadata = {
                "source": str(pdf_path),
                "document_type": "pdf",
                "pages": []
            }
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                metadata["total_pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        text_content += f"\n\n--- Page {page_num} ---\n\n"
                        text_content += page_text
                        
                        metadata["pages"].append({
                            "page_number": page_num,
                            "character_count": len(page_text),
                            "word_count": len(page_text.split())
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        metadata["pages"].append({
                            "page_number": page_num,
                            "error": str(e)
                        })
            
            logger.info(f"Extracted text from {pdf_path}: {len(text_content)} characters, {metadata['total_pages']} pages")
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_path}: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Remove extra newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove common PDF artifacts
        text = re.sub(r'\x00', '', text)  # Null characters
        text = re.sub(r'[\x80-\x9F]', '', text)  # Control characters
        
        return text
    
    def extract_tax_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract tax-specific metadata from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tax-specific metadata
        """
        tax_metadata = {
            "tax_terms": text_processor.extract_tax_terms(text),
            "numbers": text_processor.extract_numbers(text, include_currency=True),
            "dates": text_processor.extract_dates(text),
            "text_stats": text_processor.get_text_statistics(text)
        }
        
        # Extract specific tax concepts
        tax_types = []
        for term in tax_metadata["tax_terms"]:
            if term["type"] == "tax_types":
                tax_types.append(term["text"])
        tax_metadata["tax_types"] = list(set(tax_types))  # Remove duplicates
        
        # Extract rates and percentages
        rates = []
        for num in tax_metadata["numbers"]:
            if num["type"] == "percentage":
                rates.append(num["value"])
        tax_metadata["tax_rates"] = rates
        
        # Extract currency amounts
        amounts = []
        for num in tax_metadata["numbers"]:
            if num["type"] == "currency":
                amounts.append(num["value"])
        tax_metadata["currency_amounts"] = amounts
        
        return tax_metadata

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create text chunks with overlap and tax-specific metadata
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        
        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Extract tax-specific metadata for this chunk
                chunk_tax_metadata = self.extract_tax_metadata(current_chunk)
                
                # Create chunk with enhanced metadata
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),  # This will be correct since we add after validation
                        "sentence_count": len(current_sentences),
                        "character_count": len(current_chunk),
                        **chunk_tax_metadata
                    }
                )
                # Only add chunk if it passes validation
                if self._validate_chunk(chunk):
                    chunk.metadata["chunk_index"] = len(chunks)  # Set correct index
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._create_overlap(current_sentences, chunk_overlap)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                current_sentences = self._get_overlap_sentences(current_sentences, chunk_overlap) + [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk_tax_metadata = self.extract_tax_metadata(current_chunk)
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "sentence_count": len(current_sentences),
                    "character_count": len(current_chunk),
                    **chunk_tax_metadata
                }
            )
            # Only add chunk if it passes validation
            if self._validate_chunk(chunk):
                chunk.metadata["chunk_index"] = len(chunks)  # Set correct index
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document with tax-specific metadata")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        # Improved sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short sentences (likely artifacts)
        filtered_sentences = []
        for sentence in sentences:
            # Skip very short sentences unless they contain important tax info
            if len(sentence) < 10:
                if any(tax_word in sentence.lower() for tax_word in ['tax', 'b&o', '%', '$']):
                    filtered_sentences.append(sentence)
                # Otherwise skip very short sentences
            else:
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def _create_overlap(self, sentences: List[str], overlap_chars: int) -> str:
        """Create overlap text from previous sentences"""
        if not sentences:
            return ""
        
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= overlap_chars:
                overlap_text = sentence + " " + overlap_text if overlap_text else sentence
            else:
                break
        
        return overlap_text.strip()
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """Get sentences for overlap"""
        overlap_sentences = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def _validate_chunk(self, chunk: DocumentChunk) -> bool:
        """
        Validate if a chunk meets quality criteria
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            True if chunk is valid
        """
        content = chunk.content.strip()
        
        # Skip chunks that are too short
        if len(content) < 20:
            return False
            
        # Skip chunks that are mostly whitespace or single characters
        if len(content.replace(' ', '').replace('\n', '').replace('\t', '')) < 10:
            return False
            
        # Skip chunks that are just page numbers or headers
        if re.match(r'^\d+$', content) or re.match(r'^Page \d+', content, re.IGNORECASE):
            return False
            
        return True
    
    def process_pdf(self, pdf_path: Optional[str] = None) -> List[DocumentChunk]:
        """
        Process PDF file and return chunks
        
        Args:
            pdf_path: Path to PDF file (uses config default if not provided)
            
        Returns:
            List of DocumentChunk objects
        """
        pdf_path = pdf_path or self.config.pdf_path
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text, metadata = self.extract_text_from_pdf(pdf_path)
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Create chunks
        chunks = self.create_chunks(cleaned_text, metadata)
        
        # Save processed chunks
        self.save_processed_chunks(chunks)
        
        return chunks
    
    def save_processed_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Save processed chunks to disk"""
        output_path = Path(self.config.processed_docs_path) / "processed_chunks.json"
        
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} processed chunks to {output_path}")
    
    def load_processed_chunks(self) -> List[DocumentChunk]:
        """Load previously processed chunks"""
        input_path = Path(self.config.processed_docs_path) / "processed_chunks.json"
        
        if not input_path.exists():
            logger.warning(f"Processed chunks file not found: {input_path}")
            return []
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            chunks = []
            for data in chunk_data:
                chunk = DocumentChunk(
                    content=data["content"],
                    metadata=data["metadata"],
                    chunk_id=data.get("chunk_id")
                )
                chunks.append(chunk)
            
            logger.info(f"Loaded {len(chunks)} processed chunks from {input_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load processed chunks: {str(e)}")
            return []
    
    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {"total_chunks": 0}
        
        content_lengths = [len(chunk.content) for chunk in chunks]
        sentence_counts = [chunk.metadata.get("sentence_count", 0) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "average_length": sum(content_lengths) / len(content_lengths),
            "min_length": min(content_lengths),
            "max_length": max(content_lengths),
            "total_characters": sum(content_lengths),
            "average_sentences": sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
            "source_document": chunks[0].metadata.get("source", "unknown")
        }
    
    def detect_document_jurisdiction(self, pdf_path: str, text_content: str) -> str:
        """
        Detect jurisdiction for a document based on filename and content
        
        Args:
            pdf_path: Path to the PDF file
            text_content: Extracted text content
            
        Returns:
            Jurisdiction string ("california", "general", etc.)
        """
        pdf_name = Path(pdf_path).name.lower()
        
        # Check filename for jurisdiction indicators
        if "pub29" in pdf_name or "california" in pdf_name or "ca" in pdf_name:
            logger.info(f"Detected California jurisdiction from filename: {pdf_name}")
            return "california"
        
        # Check content for jurisdiction indicators (first 2000 characters)
        sample_text = text_content[:2000] if text_content else ""
        detected_jurisdiction = jurisdiction_detector.detect_jurisdiction(sample_text)
        
        if detected_jurisdiction:
            logger.info(f"Detected {detected_jurisdiction} jurisdiction from content analysis")
            return detected_jurisdiction
        
        # Default to general
        logger.info(f"No specific jurisdiction detected, defaulting to general: {pdf_name}")
        return "general"
    
    def process_pdf_with_jurisdiction(self, pdf_path: str, jurisdiction: Optional[str] = None) -> List[DocumentChunk]:
        """
        Process PDF file with jurisdiction metadata
        
        Args:
            pdf_path: Path to PDF file
            jurisdiction: Optional jurisdiction override (auto-detected if not provided)
            
        Returns:
            List of DocumentChunk objects with jurisdiction metadata
        """
        logger.info(f"Processing PDF with jurisdiction metadata: {pdf_path}")
        
        # Extract text from PDF
        text, metadata = self.extract_text_from_pdf(pdf_path)
        
        # Detect or use provided jurisdiction
        if jurisdiction is None:
            jurisdiction = self.detect_document_jurisdiction(pdf_path, text)
        
        # Add jurisdiction to document metadata
        metadata["jurisdiction"] = jurisdiction
        metadata["jurisdiction_detection"] = "auto" if jurisdiction != "general" else "default"
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Create chunks with jurisdiction metadata
        chunks = self.create_chunks_with_jurisdiction(cleaned_text, metadata, jurisdiction)
        
        # Save processed chunks
        self.save_processed_chunks_with_jurisdiction(chunks, jurisdiction)
        
        logger.info(f"Processed {len(chunks)} chunks with jurisdiction: {jurisdiction}")
        return chunks
    
    def create_chunks_with_jurisdiction(
        self, 
        text: str, 
        metadata: Dict[str, Any], 
        jurisdiction: str
    ) -> List[DocumentChunk]:
        """
        Create text chunks with jurisdiction metadata
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            jurisdiction: Document jurisdiction
            
        Returns:
            List of DocumentChunk objects with jurisdiction metadata
        """
        # Use existing chunk creation logic
        chunks = self.create_chunks(text, metadata)
        
        # Add jurisdiction metadata to each chunk
        for chunk in chunks:
            chunk.metadata["jurisdiction"] = jurisdiction
            
            # Add jurisdiction-specific metadata
            if jurisdiction == "california":
                chunk.metadata["jurisdiction_type"] = "state_specific"
                chunk.metadata["state"] = "California"
            else:
                chunk.metadata["jurisdiction_type"] = "general"
        
        logger.debug(f"Created {len(chunks)} chunks with {jurisdiction} jurisdiction metadata")
        return chunks
    
    def process_multiple_pdfs(self, pdf_configs: List[Dict[str, str]]) -> List[DocumentChunk]:
        """
        Process multiple PDFs with their respective jurisdictions
        
        Args:
            pdf_configs: List of dicts with 'path' and optional 'jurisdiction' keys
            
        Returns:
            Combined list of DocumentChunk objects
        """
        all_chunks = []
        
        for config in pdf_configs:
            pdf_path = config["path"]
            jurisdiction = config.get("jurisdiction")
            
            try:
                chunks = self.process_pdf_with_jurisdiction(pdf_path, jurisdiction)
                all_chunks.extend(chunks)
                logger.info(f"Added {len(chunks)} chunks from {Path(pdf_path).name}")
                
            except Exception as e:
                logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
                continue
        
        # Save combined chunks
        if all_chunks:
            self.save_processed_chunks(all_chunks)
            logger.info(f"Processed total of {len(all_chunks)} chunks from {len(pdf_configs)} PDFs")
        
        return all_chunks
    
    def save_processed_chunks_with_jurisdiction(self, chunks: List[DocumentChunk], jurisdiction: str) -> None:
        """Save processed chunks with jurisdiction-specific filename"""
        output_path = Path(self.config.processed_docs_path) / f"processed_chunks_{jurisdiction}.json"
        
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk.chunk_id
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} {jurisdiction} chunks to {output_path}")
    
    def get_jurisdiction_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about chunks by jurisdiction"""
        if not chunks:
            return {"total_chunks": 0}
        
        jurisdiction_counts = {}
        jurisdiction_chars = {}
        
        for chunk in chunks:
            jurisdiction = chunk.metadata.get("jurisdiction", "unknown")
            jurisdiction_counts[jurisdiction] = jurisdiction_counts.get(jurisdiction, 0) + 1
            jurisdiction_chars[jurisdiction] = jurisdiction_chars.get(jurisdiction, 0) + len(chunk.content)
        
        return {
            "total_chunks": len(chunks),
            "jurisdiction_breakdown": jurisdiction_counts,
            "jurisdiction_characters": jurisdiction_chars,
            "jurisdictions": list(jurisdiction_counts.keys())
        }


# Global document processor instance
document_processor = DocumentProcessor()