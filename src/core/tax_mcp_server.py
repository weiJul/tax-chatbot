"""
Real MCP Tax Database Server
Implements proper Model Context Protocol for tax user data access
"""
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


@dataclass
class UserData:
    """User data structure"""
    id: int
    first_name: str
    last_name: str
    email: str
    phone: str
    tax_id: str
    address: str
    city: str
    state: str
    zip_code: str
    filing_status: str
    annual_income: float
    created_at: str
    updated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class TaxDatabaseMCP:
    """Database operations for MCP server"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection"""
        if db_path is None:
            # Default to customers.db in resources/customers_data/
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "resources" / "customers_data" / "customers.db"
        
        self.db_path = Path(db_path)
        logger.info(f"Initializing MCP database with: {self.db_path}")
        
        # Verify database exists
        if not self.db_path.exists():
            logger.warning(f"Database not found at {self.db_path}")
    
    def _execute_query(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                # Convert rows to dictionaries
                results = [dict(row) for row in cursor.fetchall()]
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Database query failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in database query: {str(e)}")
            raise
    
    def get_user_by_name(self, first_name: str, last_name: str) -> Optional[UserData]:
        """Get user by first and last name"""
        # Try exact match first
        exact_query = """
            SELECT * FROM customers 
            WHERE LOWER(first_name) = LOWER(?) AND LOWER(last_name) = LOWER(?)
            LIMIT 1
        """
        
        try:
            results = self._execute_query(exact_query, (first_name, last_name))
            
            if results:
                user_data = UserData(**results[0])
                logger.debug(f"Found exact match for: {first_name} {last_name}")
                return user_data
            
            # Try partial matching
            fuzzy_query = """
                SELECT * FROM customers 
                WHERE LOWER(first_name) LIKE LOWER(?) AND LOWER(last_name) LIKE LOWER(?)
                LIMIT 1
            """
            
            fuzzy_results = self._execute_query(fuzzy_query, (f"%{first_name}%", f"%{last_name}%"))
            
            if fuzzy_results:
                user_data = UserData(**fuzzy_results[0])
                logger.debug(f"Found partial match for '{first_name} {last_name}': {user_data.first_name} {user_data.last_name}")
                return user_data
            
            logger.debug(f"No user found with name: {first_name} {last_name}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by name {first_name} {last_name}: {str(e)}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[UserData]:
        """Get user by email address"""
        query = """
            SELECT * FROM customers 
            WHERE LOWER(email) = LOWER(?)
            LIMIT 1
        """
        
        try:
            results = self._execute_query(query, (email,))
            
            if not results:
                logger.info(f"No user found with email: {email}")
                return None
            
            user_data = results[0]
            return UserData(**user_data)
            
        except Exception as e:
            logger.error(f"Failed to get user by email {email}: {str(e)}")
            return None
    
    def get_user_by_tax_id(self, tax_id: str) -> Optional[UserData]:
        """Get user by tax ID (SSN)"""
        query = """
            SELECT * FROM customers 
            WHERE tax_id = ?
            LIMIT 1
        """
        
        try:
            results = self._execute_query(query, (tax_id,))
            
            if not results:
                logger.info(f"No user found with tax ID: {tax_id}")
                return None
            
            user_data = results[0]
            return UserData(**user_data)
            
        except Exception as e:
            logger.error(f"Failed to get user by tax ID {tax_id}: {str(e)}")
            return None
    
    def search_users(self, search_term: str) -> List[UserData]:
        """Search users by name, email, or tax ID"""
        query = """
            SELECT * FROM customers 
            WHERE LOWER(first_name) LIKE LOWER(?) 
            OR LOWER(last_name) LIKE LOWER(?)
            OR LOWER(email) LIKE LOWER(?)
            OR tax_id LIKE ?
            LIMIT 10
        """
        
        search_pattern = f"%{search_term}%"
        
        try:
            results = self._execute_query(query, (search_pattern, search_pattern, search_pattern, search_pattern))
            
            users = []
            for user_data in results:
                users.append(UserData(**user_data))
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to search users with term {search_term}: {str(e)}")
            return []
    
    def get_all_users(self) -> List[UserData]:
        """Get all users"""
        query = "SELECT * FROM customers ORDER BY last_name, first_name"
        
        try:
            results = self._execute_query(query)
            
            users = []
            for user_data in results:
                users.append(UserData(**user_data))
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to get all users: {str(e)}")
            return []
    
    def calculate_tax_bracket(self, annual_income: float) -> str:
        """Calculate simplified tax bracket based on 2023 rates (single filer)"""
        if annual_income <= 11000:
            return "10%"
        elif annual_income <= 44725:
            return "12%"
        elif annual_income <= 95375:
            return "22%"
        elif annual_income <= 182050:
            return "24%"
        elif annual_income <= 231250:
            return "32%"
        elif annual_income <= 578125:
            return "35%"
        else:
            return "37%"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database connection status and stats"""
        try:
            if not self.db_path.exists():
                return {
                    'status': 'error',
                    'message': f'Database file not found: {self.db_path}'
                }
            
            # Test connection with simple query
            results = self._execute_query("SELECT COUNT(*) as count FROM customers")
            user_count = results[0]['count'] if results else 0
            
            return {
                'status': 'connected',
                'database_path': str(self.db_path),
                'user_count': user_count,
                'message': 'Database connection successful'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Database connection failed: {str(e)}'
            }


# Initialize MCP server
mcp = FastMCP("tax-database-server")

# Initialize database connection
db = TaxDatabaseMCP()


# MCP Resources - for retrieving user data
@mcp.resource("user://profile/{user_id}")
async def get_user_profile(user_id: str) -> str:
    """Get user profile by ID"""
    try:
        user_id_int = int(user_id)
        # Get user by direct query
        query = "SELECT * FROM customers WHERE id = ?"
        results = db._execute_query(query, (user_id_int,))
        
        if not results:
            return f"User with ID {user_id} not found"
        
        user = UserData(**results[0])
        return f"""User Profile:
Name: {user.first_name} {user.last_name}
Email: {user.email}
Tax ID: {user.tax_id}
Filing Status: {user.filing_status}
Annual Income: ${user.annual_income:,.2f}
State: {user.state}
Tax Bracket: {db.calculate_tax_bracket(user.annual_income)}
"""
        
    except ValueError:
        return f"Invalid user ID: {user_id}"
    except Exception as e:
        return f"Error retrieving user profile: {str(e)}"


@mcp.resource("user://tax-context/{user_id}")
async def get_user_tax_context(user_id: str) -> str:
    """Get user's tax-relevant data formatted for LLM context"""
    try:
        user_id_int = int(user_id)
        query = """
            SELECT first_name, last_name, filing_status, annual_income, state 
            FROM customers 
            WHERE id = ?
        """
        
        results = db._execute_query(query, (user_id_int,))
        
        if not results:
            return f"User with ID {user_id} not found"
        
        user_data = results[0]
        
        # Format data for LLM context
        tax_context = {
            'user_name': f"{user_data['first_name']} {user_data['last_name']}",
            'filing_status': user_data['filing_status'],
            'annual_income': user_data['annual_income'],
            'state': user_data['state'],
            'tax_bracket': db.calculate_tax_bracket(user_data['annual_income']),
            'deduction_type': 'standard' if user_data['filing_status'] in ['single', 'married_separately'] else 'married_joint'
        }
        
        return f"""Tax Context for {tax_context['user_name']}:
Filing Status: {tax_context['filing_status']}
Annual Income: ${tax_context['annual_income']:,.2f}
Tax Bracket: {tax_context['tax_bracket']}
State: {tax_context['state']}
Recommended Deduction: {tax_context['deduction_type']}
"""
        
    except ValueError:
        return f"Invalid user ID: {user_id}"
    except Exception as e:
        return f"Error retrieving tax context: {str(e)}"


# MCP Tools - for performing operations
@mcp.tool()
async def search_users_by_name(first_name: str, last_name: str) -> Dict[str, Any]:
    """Search for a user by first and last name"""
    try:
        user = db.get_user_by_name(first_name, last_name)
        
        if user:
            return {
                "found": True,
                "user": user.to_dict(),
                "tax_bracket": db.calculate_tax_bracket(user.annual_income)
            }
        else:
            return {
                "found": False,
                "message": f"No user found with name: {first_name} {last_name}"
            }
            
    except Exception as e:
        return {
            "found": False,
            "error": str(e)
        }


@mcp.tool()
async def search_users_by_email(email: str) -> Dict[str, Any]:
    """Search for a user by email address"""
    try:
        user = db.get_user_by_email(email)
        
        if user:
            return {
                "found": True,
                "user": user.to_dict(),
                "tax_bracket": db.calculate_tax_bracket(user.annual_income)
            }
        else:
            return {
                "found": False,
                "message": f"No user found with email: {email}"
            }
            
    except Exception as e:
        return {
            "found": False,
            "error": str(e)
        }


@mcp.tool()
async def search_users_by_tax_id(tax_id: str) -> Dict[str, Any]:
    """Search for a user by tax ID (SSN)"""
    try:
        user = db.get_user_by_tax_id(tax_id)
        
        if user:
            return {
                "found": True,
                "user": user.to_dict(),
                "tax_bracket": db.calculate_tax_bracket(user.annual_income)
            }
        else:
            return {
                "found": False,
                "message": f"No user found with tax ID: {tax_id}"
            }
            
    except Exception as e:
        return {
            "found": False,
            "error": str(e)
        }


@mcp.tool()
async def search_users_general(search_term: str) -> List[Dict[str, Any]]:
    """General search for users by name, email, or tax ID"""
    try:
        users = db.search_users(search_term)
        
        result = []
        for user in users:
            user_dict = user.to_dict()
            user_dict['tax_bracket'] = db.calculate_tax_bracket(user.annual_income)
            result.append(user_dict)
        
        return result
        
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def get_database_status() -> Dict[str, Any]:
    """Get database connection status and statistics"""
    try:
        return db.get_database_stats()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@mcp.tool()
async def calculate_tax_bracket_for_income(annual_income: float) -> Dict[str, Any]:
    """Calculate tax bracket for a given annual income"""
    try:
        tax_bracket = db.calculate_tax_bracket(annual_income)
        
        return {
            "annual_income": annual_income,
            "tax_bracket": tax_bracket,
            "calculation_method": "2023 single filer rates"
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }


@mcp.tool()
async def get_all_users() -> List[Dict[str, Any]]:
    """Get all users in the database"""
    try:
        users = db.get_all_users()
        
        result = []
        for user in users:
            user_dict = user.to_dict()
            user_dict['tax_bracket'] = db.calculate_tax_bracket(user.annual_income)
            result.append(user_dict)
        
        return result
        
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool()
async def get_user_tax_context(user_id: int) -> Dict[str, Any]:
    """Get user's tax context data by ID for LLM enhancement"""
    try:
        query = """
            SELECT first_name, last_name, filing_status, annual_income, state 
            FROM customers 
            WHERE id = ?
        """
        
        results = db._execute_query(query, (user_id,))
        
        if not results:
            return {
                "error": f"User with ID {user_id} not found"
            }
        
        user = results[0]
        tax_bracket = db.calculate_tax_bracket(user['annual_income'])
        
        return {
            "user_id": user_id,
            "name": f"{user['first_name']} {user['last_name']}",
            "filing_status": user['filing_status'],
            "annual_income": float(user['annual_income']),
            "tax_bracket": tax_bracket,
            "state": user['state']
        }
        
    except Exception as e:
        return {
            "error": f"Error retrieving user tax context: {str(e)}"
        }


if __name__ == "__main__":
    # Run the MCP server
    import asyncio
    
    async def main():
        # Initialize and run server
        await mcp.run()
    
    asyncio.run(main())