"""
MCP Client for Tax Database
Provides client interface to communicate with the MCP tax database server
"""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """Context object that matches the original interface"""
    found: bool
    user_data: Optional[Dict[str, Any]] = None
    tax_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'found': self.found,
            'user_data': self.user_data,
            'tax_context': self.tax_context
        }


class TaxMCPClient:
    """MCP client for tax database operations"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.connected = False
        
    async def _ensure_connected(self):
        """Ensure MCP server connection is active"""
        if not self.connected or self.session is None:
            await self._connect()
    
    async def _connect(self):
        """Connect to the MCP server"""
        try:
            # For now, we'll create a local connection
            # In a real deployment, this would connect to the actual MCP server
            logger.info("Connecting to MCP tax database server...")
            
            # Initialize server parameters for stdio connection
            # Path to the standalone MCP server script
            project_root = Path(__file__).parent.parent.parent
            server_script = project_root / "mcp_tax_server.py"
            
            server_params = StdioServerParameters(
                command="python",
                args=[str(server_script)],
                env=None
            )
            
            # Create stdio client connection using context manager
            self.stdio_context = stdio_client(server_params)
            read, write = await self.stdio_context.__aenter__()
            
            # Create client session
            self.session_context = ClientSession(read, write)
            self.session = await self.session_context.__aenter__()
            
            # Initialize the connection
            await self.session.initialize()
            
            self.connected = True
            logger.info("Successfully connected to MCP server")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            self.connected = False
            raise
    
    async def _call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call an MCP tool with parameters"""
        try:
            await self._ensure_connected()
            
            if not self.session:
                raise RuntimeError("MCP session not established")
            
            # Call the tool through MCP protocol
            result = await self.session.call_tool(tool_name, kwargs)
            
            # Extract content from MCP response
            # For tools that return structured data, prefer structuredContent
            if hasattr(result, 'structuredContent') and result.structuredContent:
                # Return structured content directly for list results
                if 'result' in result.structuredContent:
                    return result.structuredContent['result']
                return result.structuredContent
            
            # Fallback to text content for simple responses  
            if hasattr(result, 'content') and result.content:
                return result.content[0].text if result.content else None
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to call MCP tool {tool_name}: {str(e)}")
            raise
    
    async def _read_resource(self, resource_uri: str) -> str:
        """Read an MCP resource"""
        try:
            await self._ensure_connected()
            
            if not self.session:
                raise RuntimeError("MCP session not established")
            
            # Read the resource through MCP
            result = await self.session.read_resource(resource_uri)
            return result.contents[0].text if result.contents else ""
            
        except Exception as e:
            logger.error(f"Failed to read MCP resource {resource_uri}: {str(e)}")
            raise
    
    async def get_user_by_name(self, first_name: str, last_name: str) -> Optional['UserData']:
        """Get user by first and last name"""
        try:
            result = await self._call_tool("search_users_by_name", first_name=first_name, last_name=last_name)
            
            # Parse JSON result from MCP tool
            if result:
                import json
                if isinstance(result, str):
                    data = json.loads(result)
                else:
                    data = result
                    
                if data.get("found"):
                    from .tax_mcp_server import \
                        UserData  # Import for compatibility
                    return UserData(**data["user"])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by name: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional['UserData']:
        """Get user by email address"""
        try:
            result = await self._call_tool("search_users_by_email", email=email)
            
            # Parse JSON result from MCP tool
            if result:
                import json
                if isinstance(result, str):
                    data = json.loads(result)
                else:
                    data = result
                    
                if data.get("found"):
                    from .tax_mcp_server import UserData
                    return UserData(**data["user"])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by email: {str(e)}")
            return None
    
    async def get_user_by_tax_id(self, tax_id: str) -> Optional['UserData']:
        """Get user by tax ID"""
        try:
            result = await self._call_tool("search_users_by_tax_id", tax_id=tax_id)
            
            # Parse JSON result from MCP tool
            if result:
                import json
                if isinstance(result, str):
                    data = json.loads(result)
                else:
                    data = result
                    
                if data.get("found"):
                    from .tax_mcp_server import UserData
                    return UserData(**data["user"])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by tax ID: {str(e)}")
            return None
    
    async def search_users(self, search_term: str) -> List['UserData']:
        """Search users by general term"""
        try:
            result = await self._call_tool("search_users_general", search_term=search_term)
            
            # Parse JSON result from MCP tool
            if result:
                import json
                if isinstance(result, str):
                    data = json.loads(result)
                else:
                    data = result
                    
                if isinstance(data, list):
                    from .tax_mcp_server import UserData
                    users = []
                    for user_data in data:
                        if "error" not in user_data:
                            users.append(UserData(**user_data))
                    return users
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to search users: {str(e)}")
            return []
    
    async def get_user_tax_data(self, user_id: int) -> Dict[str, Any]:
        """Get user's tax context by ID"""
        try:
            # Use tool instead of resource to avoid event loop issues
            result = await self._call_tool("get_user_tax_context", user_id=user_id)
            
            # Handle structured data from MCP tool
            if result:
                # Check for error in response
                if isinstance(result, dict) and 'error' in result:
                    logger.error(f"MCP tool error: {result['error']}")
                    return {}
                    
                return result if isinstance(result, dict) else {}
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get tax data: {str(e)}")
            return {}
    
    async def check_database_connection(self) -> Dict[str, Any]:
        """Check database connection status"""
        try:
            result = await self._call_tool("get_database_status")
            
            # Parse JSON result from MCP tool
            if result:
                import json
                if isinstance(result, str):
                    return json.loads(result)
                else:
                    return result
            
            return {"status": "error", "message": "No response from MCP server"}
            
        except Exception as e:
            logger.error(f"Failed to check database connection: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_all_users(self) -> List['UserData']:
        """Get all users using dedicated MCP tool"""
        try:
            result = await self._call_tool("get_all_users")
            
            # Handle structured data from MCP tool
            if result and isinstance(result, list):
                from .tax_mcp_server import UserData
                users = []
                for user_data in result:
                    if "error" not in user_data:
                        # Remove calculated fields that aren't part of UserData schema
                        clean_data = {k: v for k, v in user_data.items() if k != 'tax_bracket'}
                        users.append(UserData(**clean_data))
                return users
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get all users: {str(e)}")
            return []
    
    async def disconnect(self):
        """Disconnect from MCP server"""
        try:
            if hasattr(self, 'session_context') and self.session_context:
                await self.session_context.__aexit__(None, None, None)
            if hasattr(self, 'stdio_context') and self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {str(e)}")
        finally:
            self.session = None
            self.session_context = None
            self.stdio_context = None
            self.connected = False


# Pure MCP compatibility layer - provides synchronous interface using only MCP protocol
class MCPServerCompatibility:
    """
    Pure MCP compatibility layer that provides synchronous interface matching the original mcp_server module.
    This allows existing code to work with minimal changes while using ONLY real MCP protocol underneath.
    No fallbacks - all operations go through MCP client.
    """
    
    def __init__(self):
        self._connection_established = False
        
    def _run_async_pure(self, coro_func):
        """Run async function in sync context - pure MCP only"""
        try:
            # Create fresh client and event loop for each operation to avoid session reuse issues
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Create fresh client instance to avoid session state issues
                client = TaxMCPClient()
                result = loop.run_until_complete(coro_func(client))
                self._connection_established = True
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"MCP operation failed: {str(e)}")
            self._connection_established = False
            # Return appropriate empty result based on expected return type
            return None
    
    def get_user_by_name(self, first_name: str, last_name: str):
        """Get user by name - pure MCP only"""
        logger.debug(f"MCP: Getting user by name: {first_name} {last_name}")
        return self._run_async_pure(lambda client: client.get_user_by_name(first_name, last_name))
    
    def get_user_by_email(self, email: str):
        """Get user by email - pure MCP only"""
        logger.debug(f"MCP: Getting user by email: {email}")
        return self._run_async_pure(lambda client: client.get_user_by_email(email))
    
    def get_user_by_tax_id(self, tax_id: str):
        """Get user by tax ID - pure MCP only"""
        logger.debug(f"MCP: Getting user by tax ID: {tax_id}")
        return self._run_async_pure(lambda client: client.get_user_by_tax_id(tax_id))
    
    def search_users(self, search_term: str):
        """Search users - pure MCP only"""
        logger.debug(f"MCP: Searching users: {search_term}")
        result = self._run_async_pure(lambda client: client.search_users(search_term))
        return result if result is not None else []
    
    def get_user_tax_data(self, user_id: int):
        """Get user tax data - pure MCP only"""
        logger.debug(f"MCP: Getting tax data for user ID: {user_id}")
        result = self._run_async_pure(lambda client: client.get_user_tax_data(user_id))
        return result if result is not None else {}
    
    def check_database_connection(self):
        """Check database connection - pure MCP only"""
        logger.debug("MCP: Checking database connection")
        result = self._run_async_pure(lambda client: client.check_database_connection())
        if result is None:
            return {"status": "error", "message": "MCP server unavailable"}
        return result
    
    def get_all_users(self):
        """Get all users - pure MCP only"""
        logger.debug("MCP: Getting all users")
        result = self._run_async_pure(lambda client: client.get_all_users())
        return result if result is not None else []


# Create global instance for backward compatibility
mcp_client_compat = MCPServerCompatibility()

# For gradual migration - can be imported as mcp_server for drop-in replacement
mcp_server = mcp_client_compat