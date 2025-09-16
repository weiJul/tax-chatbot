# MCP Server API Documentation

## Overview

The Tax RAG System includes a **Model Context Protocol (MCP) Server** that provides standardized access to user tax data. This server implements the MCP specification for secure, isolated database operations and can be used by external MCP-compatible clients like Claude Desktop.

## Architecture

### Pure MCP Implementation
- **Process Isolation**: Database operations run in separate process
- **JSON-RPC Protocol**: Standard MCP communication over stdio transport
- **No Fallbacks**: Pure protocol compliance with graceful error handling
- **Standards Compliant**: Compatible with any MCP-compatible client

### Transport Layer
```
Client Application ←→ JSON-RPC over stdio ←→ MCP Server ←→ SQLite Database
```

## Server Setup

### Starting the MCP Server

```bash
# Standalone server (recommended for external clients)
python mcp_tax_server.py

# Integrated server (used by CLI and web interfaces)
# Automatically started by tax_mcp_client.py when needed
```

### Configuration

The MCP server is configured via `config.yaml`:

```yaml
mcp:
  server:
    script_path: "./mcp_tax_server.py"
    transport: "stdio"
    timeout_seconds: 30
    startup_wait: 2.0
  
  client:
    connection_retry_delay: 1.0
    max_connection_attempts: 3
    connection_timeout: 10.0
    enable_pure_mcp_only: true
  
  error_handling:
    return_empty_on_failure: true
    log_mcp_errors: true
    fail_gracefully: true
```

## MCP Resources

The server provides standardized MCP resources for user data access:

### Resource Templates

#### User Profile Resource
- **URI Template**: `user://profile/{user_id}`
- **Description**: Complete user profile including personal and tax information
- **Content Type**: `application/json`

#### Tax Context Resource  
- **URI Template**: `user://tax-context/{user_id}`
- **Description**: Tax-specific context for RAG enhancement
- **Content Type**: `application/json`

### Resource Operations

#### Read User Profile
```json
{
  "method": "resources/read",
  "params": {
    "uri": "user://profile/1"
  }
}
```

**Response:**
```json
{
  "contents": [
    {
      "uri": "user://profile/1",
      "mimeType": "application/json", 
      "text": "{\"id\": 1, \"first_name\": \"Sarah\", \"last_name\": \"Johnson\", ...}"
    }
  ]
}
```

## MCP Tools

The server provides several tools for user data lookup and management:

### 1. Search User by Name

**Tool ID**: `search_user_by_name`

**Description**: Search for users by first and last name with fuzzy matching

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "first_name": {
      "type": "string",
      "description": "User's first name"
    },
    "last_name": {
      "type": "string", 
      "description": "User's last name"
    }
  },
  "required": ["first_name", "last_name"]
}
```

**Example Call**:
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_user_by_name",
    "arguments": {
      "first_name": "Sarah",
      "last_name": "Johnson"
    }
  }
}
```

**Example Response**:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"found\": true, \"user\": {\"id\": 1, \"first_name\": \"Sarah\", \"last_name\": \"Johnson\", \"email\": \"sarah.johnson@email.com\", \"tax_id\": \"123-45-6789\", \"filing_status\": \"single\", \"annual_income\": 75000.00, \"state\": \"TX\"}}"
    }
  ]
}
```

### 2. Search User by Email

**Tool ID**: `search_user_by_email`

**Description**: Find user by email address

**Parameters**:
```json
{
  "type": "object", 
  "properties": {
    "email": {
      "type": "string",
      "description": "User's email address"
    }
  },
  "required": ["email"]
}
```

**Example Call**:
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_user_by_email", 
    "arguments": {
      "email": "sarah.johnson@email.com"
    }
  }
}
```

### 3. Search User by Tax ID

**Tool ID**: `search_user_by_tax_id`

**Description**: Find user by tax ID/SSN

**Parameters**:
```json
{
  "type": "object",
  "properties": {
    "tax_id": {
      "type": "string",
      "description": "User's tax ID or SSN"
    }
  },
  "required": ["tax_id"]
}
```

**Example Call**:
```json
{
  "method": "tools/call",
  "params": {
    "name": "search_user_by_tax_id",
    "arguments": {
      "tax_id": "123-45-6789"
    }
  }
}
```

### 4. Get User Statistics

**Tool ID**: `get_user_stats`

**Description**: Get database statistics and user count

**Parameters**: None

**Example Call**:
```json
{
  "method": "tools/call", 
  "params": {
    "name": "get_user_stats",
    "arguments": {}
  }
}
```

**Example Response**:
```json
{
  "content": [
    {
      "type": "text",
      "text": "{\"user_count\": 3, \"database_status\": \"connected\", \"last_updated\": \"2025-09-16T12:34:56\"}"
    }
  ]
}
```

## Data Models

### UserData Structure

```python
@dataclass
class UserData:
    id: int                    # Primary key
    first_name: str           # User's first name  
    last_name: str            # User's last name
    email: str                # Email address (unique)
    phone: str                # Phone number
    tax_id: str               # Tax ID/SSN (unique)
    address: str              # Street address
    city: str                 # City
    state: str                # State abbreviation
    zip_code: str             # ZIP code
    filing_status: str        # Tax filing status
    annual_income: float      # Annual income amount
    created_at: str           # Creation timestamp
    updated_at: str           # Last update timestamp
```

### Tax Context Structure (for RAG Enhancement)

```json
{
  "user_name": "Sarah Johnson",
  "filing_status": "single", 
  "annual_income": 75000.00,
  "tax_bracket": "22%",
  "state": "TX",
  "tax_id_last_4": "6789",
  "estimated_federal_tax": 16500.00,
  "estimated_state_tax": 0.00
}
```

## Database Schema

### customers table

```sql
CREATE TABLE customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    tax_id TEXT UNIQUE NOT NULL,
    address TEXT,
    city TEXT,
    state TEXT,
    zip_code TEXT,
    filing_status TEXT NOT NULL,
    annual_income DECIMAL(12,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_tax_id ON customers(tax_id);
CREATE INDEX idx_customers_name ON customers(first_name, last_name);
```

## Error Handling

### MCP Error Responses

#### User Not Found
```json
{
  "content": [
    {
      "type": "text", 
      "text": "{\"found\": false, \"error\": \"User not found\"}"
    }
  ]
}
```

#### Database Connection Error
```json
{
  "error": {
    "code": -32603,
    "message": "Internal error: Database connection failed"
  }
}
```

#### Invalid Parameters
```json
{
  "error": {
    "code": -32602,
    "message": "Invalid params: Missing required parameter 'email'"
  }
}
```

## Client Integration

### Using with Tax RAG System

The MCP server is automatically used by the Tax RAG system's query router:

```python
from src.core.tax_mcp_client import mcp_server

# Check server status
status = mcp_server.check_database_connection()
print(f"MCP Status: {status['status']} - {status['user_count']} users")

# Search for user
user = mcp_server.get_user_by_name("Sarah", "Johnson")
if user:
    print(f"Found: {user.first_name} {user.last_name} - {user.email}")
```

### Using with External MCP Clients

#### Claude Desktop Configuration
Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "tax-database": {
      "command": "python",
      "args": ["/path/to/tax-rag/mcp_tax_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/tax-rag"
      }
    }
  }
}
```

#### Python MCP Client Example
```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_tax_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # Call tool
            result = await session.call_tool(
                "search_user_by_name",
                {"first_name": "Sarah", "last_name": "Johnson"}
            )
            print(result.content[0].text)

asyncio.run(main())
```

## Security Considerations

### Data Protection
- **Local Database**: User data stored locally in SQLite database
- **Process Isolation**: MCP server runs in separate process
- **No External Network**: Server only accepts stdio connections
- **File Permissions**: Database file should have restrictive permissions (`600`)

### Access Control
- **Read-Only Operations**: Server only provides read access to user data
- **Input Validation**: All parameters validated before database queries
- **SQL Injection Prevention**: Parameterized queries used exclusively
- **Error Handling**: Detailed errors logged but not exposed to clients

### Compliance
- **Educational Use**: System designed for educational/research purposes
- **Real Data**: Use appropriate data protection measures for production
- **Audit Trail**: All database operations are logged
- **Data Retention**: Follow applicable data retention policies

## Testing & Validation

### Unit Tests

```bash
# Test MCP server functionality
python test_pure_mcp.py

# Test database operations
python -c "from src.core.tax_mcp_client import mcp_server; status = mcp_server.check_database_connection(); print(status)"

# Test user lookup
python -c "from src.core.tax_mcp_client import mcp_server; user = mcp_server.get_user_by_name('Sarah', 'Johnson'); print(user)"
```

### Integration Tests

```bash
# Test with CLI interface
python src/interfaces/cli_chat.py
# Try: "What are Sarah Johnson's tax obligations?"

# Test with web interface  
streamlit run src/interfaces/web_interface.py
# Try personal queries in sidebar
```

### Manual MCP Protocol Testing

```bash
# Start server manually
python mcp_tax_server.py

# Send JSON-RPC requests (in another terminal)
echo '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "search_user_by_name", "arguments": {"first_name": "Sarah", "last_name": "Johnson"}}, "id": 1}' | python mcp_tax_server.py
```

## Performance & Monitoring

### Metrics
- **Query Response Time**: Typical < 50ms for user lookups
- **Connection Success Rate**: Target > 99%
- **Database Availability**: Monitored via health checks
- **Memory Usage**: Lightweight (~10MB process)

### Monitoring Integration

The MCP server integrates with the Phoenix monitoring system:

```python
# Phoenix traces are automatically generated for MCP operations
from src.core.phoenix_tracer import get_phoenix_tracer

tracer = get_phoenix_tracer()
# MCP calls are automatically traced when Phoenix is enabled
```

### Troubleshooting

#### Common Issues

1. **Database Not Found**
   ```bash
   # Check database path
   ls -la resources/customers_data/customers.db
   
   # Recreate if missing
   cd resources/customers_data && python create_database.py
   ```

2. **Connection Timeout**
   ```bash
   # Check server status
   ps aux | grep mcp_tax_server.py
   
   # Increase timeout in config.yaml
   # mcp.client.connection_timeout: 30.0
   ```

3. **Permission Errors**
   ```bash
   # Fix database permissions
   chmod 600 resources/customers_data/customers.db
   chmod 755 resources/customers_data/
   ```

4. **Process Communication Issues**
   ```bash
   # Kill existing server processes
   pkill -f mcp_tax_server.py
   
   # Restart with debug logging
   PYTHONPATH=. python mcp_tax_server.py
   ```

---

## Reference Implementation

For a complete working example, see:
- **MCP Server**: `src/core/tax_mcp_server.py`
- **MCP Client**: `src/core/tax_mcp_client.py` 
- **Standalone Server**: `mcp_tax_server.py`
- **Integration Tests**: `test_pure_mcp.py`

This implementation follows the MCP specification and provides a production-ready foundation for secure user data access in AI applications.