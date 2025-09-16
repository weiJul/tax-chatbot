#!/usr/bin/env python3
"""
Personal Tax Query Examples

This script demonstrates personal tax queries using the fictional demo users.
These queries showcase the system's ability to route personal queries and 
enhance responses with user context from the database.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.llm_service import llm_service
from src.core.tax_mcp_client import mcp_server


def show_demo_users():
    """Display available demo users."""
    print("📋 Available Demo Users:")
    print("-" * 30)
    
    try:
        # Check database connection
        status = mcp_server.check_database_connection()
        if status['status'] == 'connected':
            print(f"✅ Database connected - {status['user_count']} users available")
            
            # Show demo user information
            demo_users = [
                ("Sarah", "Johnson", "sarah.johnson@email.com", "123-45-6789"),
                ("Michael", "Chen", "michael.chen@email.com", "987-65-4321"), 
                ("Emily", "Rodriguez", "emily.rodriguez@email.com", "456-78-9123")
            ]
            
            for first, last, email, tax_id in demo_users:
                user = mcp_server.get_user_by_name(first, last)
                if user:
                    print(f"• {user.first_name} {user.last_name}")
                    print(f"  Email: {user.email}")
                    print(f"  Tax ID: {user.tax_id}")
                    print(f"  Filing Status: {user.filing_status}")
                    print(f"  Annual Income: ${user.annual_income:,.2f}")
                    print()
                
        else:
            print(f"❌ Database connection failed: {status.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error checking demo users: {e}")


def run_personal_queries():
    """Run personal tax queries using demo users."""
    
    print("🚀 Tax Chatbot - Personal Query Examples")
    print("=" * 50)
    
    # Show available demo users
    show_demo_users()
    
    # Load the LLM service
    print("Loading models...")
    llm_service.load_model()
    print("✅ Models loaded successfully\n")
    
    # Define personal queries for demo users
    queries = [
        {
            "query": "What are Sarah Johnson's tax obligations?",
            "description": "Personal tax query using full name"
        },
        {
            "query": "Show me Michael Chen's filing status",
            "description": "User information lookup by name"
        },
        {
            "query": "What is emily.rodriguez@email.com's tax bracket?",
            "description": "User lookup by email address"
        },
        {
            "query": "My name is Sarah Johnson and I need my tax information",
            "description": "First-person personal query"
        },
        {
            "query": "Calculate taxes for tax ID 987-65-4321",
            "description": "Tax calculation using Tax ID"
        }
    ]
    
    # Process each personal query
    for i, example in enumerate(queries, 1):
        print(f"Example {i}: {example['description']}")
        print(f"Query: {example['query']}")
        print("-" * 40)
        
        try:
            # Use the router-based query method
            result = llm_service.query_with_router(example['query'])
            
            print(f"Response Type: {result.get('query_type', 'unknown')}")
            print(f"User Found: {bool(result.get('user_context') and result['user_context'].found)}")
            
            if result.get('user_context') and result['user_context'].found:
                user_info = result['user_context'].tax_context
                print(f"User: {user_info.get('user_name', 'N/A')}")
                print(f"Filing Status: {user_info.get('filing_status', 'N/A')}")
                print(f"Annual Income: ${user_info.get('annual_income', 0):,.2f}")
            
            print(f"Documents Retrieved: {result.get('num_retrieved', 'N/A')}")
            print(f"Response:\n{result['response']}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("\n" + "=" * 50 + "\n")
    
    print("✅ Personal query examples completed!")


if __name__ == "__main__":
    run_personal_queries()