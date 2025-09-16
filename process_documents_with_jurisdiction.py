#!/usr/bin/env python3
"""
Document Processing Script for Jurisdiction-Aware Tax RAG System
Processes both general and California-specific tax documents with proper metadata
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import logging
from typing import List, Dict, Any

from src.core.document_processor import document_processor
from src.core.vector_store import vector_store
from src.core.embeddings import embedding_service
from src.utils.memory_monitor import memory_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main processing function"""
    logger.info("Starting jurisdiction-aware document processing...")
    
    try:
        # Configuration for both documents
        pdf_configs = [
            {
                "path": "./resources/information_material/BusinessTaxBasics_0.pdf",
                "jurisdiction": "general",
                "description": "General business tax regulations and procedures"
            },
            {
                "path": "./resources/information_material/pub29.pdf", 
                "jurisdiction": "california",
                "description": "California-specific tax regulations (Publication 29)"
            }
        ]
        
        # Process all documents
        logger.info(f"Processing {len(pdf_configs)} documents...")
        all_chunks = []
        
        for config in pdf_configs:
            pdf_path = config["path"]
            jurisdiction = config["jurisdiction"]
            description = config["description"]
            
            logger.info(f"\n--- Processing {description} ---")
            logger.info(f"File: {pdf_path}")
            logger.info(f"Jurisdiction: {jurisdiction}")
            
            if not Path(pdf_path).exists():
                logger.error(f"File not found: {pdf_path}")
                continue
            
            try:
                # Process document with jurisdiction metadata
                chunks = document_processor.process_pdf_with_jurisdiction(pdf_path, jurisdiction)
                all_chunks.extend(chunks)
                
                logger.info(f"✅ Processed {len(chunks)} chunks from {Path(pdf_path).name}")
                
                # Show sample chunk metadata
                if chunks:
                    sample_chunk = chunks[0]
                    logger.info(f"Sample metadata: {dict(list(sample_chunk.metadata.items())[:5])}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_path}: {str(e)}")
                continue
        
        # Display processing statistics
        logger.info(f"\n--- Processing Complete ---")
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        if all_chunks:
            # Get jurisdiction breakdown
            jurisdiction_stats = document_processor.get_jurisdiction_stats(all_chunks)
            logger.info(f"Jurisdiction breakdown: {jurisdiction_stats['jurisdiction_breakdown']}")
            
            # Save all chunks together
            document_processor.save_processed_chunks(all_chunks)
            logger.info("✅ Saved combined chunks to processed_chunks.json")
            
            # Initialize embedding service
            logger.info("\n--- Initializing Embedding Service ---")
            embedding_service.load_model()
            logger.info("✅ BGE embedding model loaded")
            
            # Connect to vector store
            logger.info("\n--- Connecting to Vector Store ---")
            vector_store.connect()
            logger.info("✅ Connected to ChromaDB")
            
            # Clear existing collection to avoid duplicates
            logger.info("🗑️  Clearing existing collection...")
            vector_store.reset_collection()
            
            # Process chunks for vector storage
            logger.info("\n--- Generating Embeddings and Storing ---")
            
            documents = []
            embeddings_list = []
            metadata_list = []
            
            batch_size = 32
            total_chunks = len(all_chunks)
            
            for i in range(0, total_chunks, batch_size):
                batch_chunks = all_chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
                
                # Generate embeddings for batch
                batch_embeddings = embedding_service.embed_documents(batch_texts)
                
                # Prepare data
                documents.extend(batch_texts)
                embeddings_list.extend(batch_embeddings)
                metadata_list.extend([chunk.metadata for chunk in batch_chunks])
                
                # Clear GPU cache periodically
                if (i // batch_size) % 3 == 0:
                    memory_monitor.clear_gpu_cache()
            
            # Store all embeddings
            logger.info(f"Storing {len(documents)} documents in vector store...")
            
            vector_store.add_documents(
                documents=documents,
                embeddings=embeddings_list,
                metadata=metadata_list
            )
            
            logger.info("✅ All documents stored in vector database")
            
            # Verify storage
            stats = vector_store.get_collection_stats()
            logger.info(f"Vector store stats: {stats}")
            
            # Show jurisdiction verification
            logger.info("\n--- Jurisdiction Verification ---")
            for jurisdiction in ["california", "general"]:
                # Simple test query to verify jurisdiction filtering
                try:
                    test_query = "tax rates"
                    test_embedding = embedding_service.embed_query(test_query)
                    
                    # Query with jurisdiction filter
                    docs, metadata, sims = vector_store.search(
                        query_embedding=test_embedding,
                        top_k=3,
                        where_filter={"jurisdiction": jurisdiction}
                    )
                    
                    logger.info(f"{jurisdiction.capitalize()}: Found {len(docs)} documents")
                    
                except Exception as e:
                    logger.error(f"Failed to verify {jurisdiction} jurisdiction: {str(e)}")
            
        else:
            logger.error("❌ No chunks were successfully processed")
        
        logger.info("\n🎉 Document processing with jurisdiction metadata complete!")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        raise
    
    finally:
        # Clean up GPU memory
        memory_monitor.clear_gpu_cache()


if __name__ == "__main__":
    main()