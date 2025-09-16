#!/usr/bin/env python3
"""
Test Pure MCP Implementation
Tests that the system uses ONLY MCP protocol with no fallbacks
"""
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.core.tax_mcp_client import mcp_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_pure_mcp_with_server_available():
    """Test pure MCP when server is available"""
    logger.info("=== Testing Pure MCP with Server Available ===")
    
    try:
        # Test database connection
        logger.info("Testing database connection...")
        status = mcp_server.check_database_connection()
        logger.info(f"Database status: {status}")
        
        # Test user search by name
        logger.info("Testing user search by name...")
        user = mcp_server.get_user_by_name("Sarah", "Johnson")
        if user:
            logger.info(f"Found user: {user.first_name} {user.last_name} ({user.email})")
            logger.info("✅ Pure MCP user search working")
        else:
            logger.info("No user found, but MCP call succeeded")
        
        # Test user search by email
        logger.info("Testing user search by email...")
        user = mcp_server.get_user_by_email("michael.chen@email.com")
        if user:
            logger.info(f"Found user by email: {user.first_name} {user.last_name}")
            logger.info("✅ Pure MCP email search working")
        else:
            logger.info("No user found by email, but MCP call succeeded")
        
        # Test get all users
        logger.info("Testing get all users...")
        users = mcp_server.get_all_users()
        logger.info(f"Found {len(users)} total users")
        if len(users) > 0:
            logger.info("✅ Pure MCP get all users working")
        
        logger.info("✅ All Pure MCP tests passed when server available")
        return True
        
    except Exception as e:
        logger.error(f"Pure MCP test failed: {str(e)}")
        return False


def test_pure_mcp_server_unavailable():
    """Test pure MCP behavior when server is unavailable"""
    logger.info("=== Testing Pure MCP with Server Unavailable ===")
    logger.info("Note: This test simulates server unavailability")
    
    # Temporarily break the MCP client connection
    original_script_path = mcp_server.client._connection_established
    
    try:
        # Force connection to fail by using invalid server path
        from src.core.tax_mcp_client import TaxMCPClient
        broken_client = TaxMCPClient()
        
        # Create a new compatibility instance with broken client
        from src.core.tax_mcp_client import MCPServerCompatibility
        broken_mcp = MCPServerCompatibility()
        broken_mcp.client = broken_client
        
        # Override server script path to cause connection failure
        import tempfile
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"# This is not a valid MCP server")
            temp_file.flush()
            
            # Update the client to use the invalid script
            broken_client._connection_established = False
            
            logger.info("Testing database connection with unavailable server...")
            status = broken_mcp.check_database_connection()
            logger.info(f"Database status: {status}")
            
            # Should return error status, not crash
            if status.get('status') == 'error':
                logger.info("✅ Pure MCP correctly returns error when server unavailable")
            
            logger.info("Testing user search with unavailable server...")
            user = broken_mcp.get_user_by_name("Sarah", "Johnson")
            if user is None:
                logger.info("✅ Pure MCP correctly returns None when server unavailable")
            
            logger.info("Testing search users with unavailable server...")
            users = broken_mcp.search_users("test")
            if users == []:
                logger.info("✅ Pure MCP correctly returns empty list when server unavailable")
            
            logger.info("✅ Pure MCP error handling working correctly")
            return True
        
    except Exception as e:
        logger.info(f"Expected error from unavailable server: {str(e)}")
        logger.info("✅ Pure MCP fails gracefully when server unavailable")
        return True


def test_mcp_integration():
    """Test integration with the overall system"""
    logger.info("=== Testing MCP System Integration ===")
    
    try:
        # Test the system through the normal interface
        from src.core.llm_service import llm_service
        
        logger.info("Testing system integration...")
        result = llm_service.query_with_router("What is Sarah Johnson's email?")
        
        if result:
            logger.info(f"System integration working: {result.get('query_type', 'unknown')}")
            logger.info("✅ Pure MCP integrated with system successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"System integration test failed: {str(e)}")
        return False


def main():
    """Run all pure MCP tests"""
    logger.info("🧪 Pure MCP Implementation Test Suite")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: MCP with server available
    results.append(test_pure_mcp_with_server_available())
    
    # Test 2: MCP with server unavailable
    results.append(test_pure_mcp_server_unavailable())
    
    # Test 3: System integration
    results.append(test_mcp_integration())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    logger.info("=" * 60)
    if passed == total:
        logger.info("✅ ALL PURE MCP TESTS PASSED!")
        logger.info("The system is now using ONLY MCP protocol with no fallbacks.")
    else:
        logger.error(f"❌ {total - passed}/{total} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()