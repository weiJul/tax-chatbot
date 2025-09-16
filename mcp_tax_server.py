#!/usr/bin/env python3
"""
Standalone MCP Tax Database Server
Runs as a separate process and communicates via stdio
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.core.tax_mcp_server import mcp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),  # Log to stderr to avoid interfering with stdio protocol
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the MCP tax database server"""
    try:
        logger.info("Starting MCP Tax Database Server...")
        
        # The FastMCP server will automatically handle stdio transport
        # Use the run() method which defaults to stdio transport
        mcp.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the server directly
    main()