#!/usr/bin/env python3
"""
Test script for router-based tax chatbot functionality
Tests both general and personal tax queries
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.query_router import query_router
from src.core.tax_mcp_client import mcp_server

def test_routing():
    """Test query routing functionality"""
    print("🔍 Testing Tax Chatbot Router")
    print("=" * 50)
    
    # Test cases
    test_queries = [
        # General tax queries
        ("What is B&O tax?", "general_tax"),
        ("When are tax deadlines?", "general_tax"),
        ("How do I file taxes in Washington State?", "general_tax"),
        
        # Personal tax queries
        ("What are Sarah Johnson's tax obligations?", "user_data"),
        ("Show me Michael Chen's filing status", "user_data"),
        ("What is my tax bracket for emily.rodriguez@email.com?", "user_data"),
    ]
    
    print("\n📊 Routing Test Results:")
    print("-" * 50)
    
    correct_predictions = 0
    total_tests = len(test_queries)
    
    for query, expected_type in test_queries:
        result = query_router.route_query(query)
        
        # Debug output - show what we actually got
        print(f"DEBUG: Router result for '{query}': {result}")
        
        predicted_type = result.get('query_type', 'unknown')
        
        # Check if prediction matches expected
        is_correct = predicted_type == expected_type
        if is_correct:
            correct_predictions += 1
        
        # Display result
        status = "✅" if is_correct else "❌"
        print(f"{status} Query: '{query}'")
        print(f"   Expected: {expected_type}, Got: {predicted_type}")
        
        # Only show confidence if it exists
        if 'confidence' in result:
            print(f"   Confidence: {result['confidence']:.2f}")
        else:
            print(f"   Confidence: Not available")
        
        # Show user context for user_data queries
        if predicted_type == "user_data":
            if result.get('user_context') and result['user_context'].found:
                user_data = result['user_context'].user_data
                print(f"   👤 User Found: {user_data.first_name} {user_data.last_name}")
                print(f"   📧 Email: {user_data.email}")
                print(f"   💰 Income: ${user_data.annual_income:,.2f}")
            else:
                error = result.get('error', 'User not found')
                print(f"   ⚠️  {error}")
        
        print()
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print(f"🎯 Routing Accuracy: {correct_predictions}/{total_tests} ({accuracy:.1f}%)")
    
    return accuracy >= 80  # Pass if 80% or better accuracy


def test_database():
    """Test database functionality"""
    print("\n💾 Testing Database Functionality")
    print("=" * 50)
    
    # Check connection
    status = mcp_server.check_database_connection()
    print(f"Database Status: {status['status']}")
    print(f"User Count: {status['user_count']}")
    
    # Test user searches
    test_searches = [
        ("Sarah", "Johnson"),
        ("michael.chen@email.com", None),
        ("nonexistent", "user")
    ]
    
    print("\n🔍 User Search Tests:")
    print("-" * 30)
    
    for search_term, last_name in test_searches:
        if last_name:
            # Name search
            user = mcp_server.get_user_by_name(search_term, last_name)
            if user:
                print(f"✅ Found: {user.first_name} {user.last_name} ({user.email})")
            else:
                print(f"❌ Not found: {search_term} {last_name}")
        else:
            # Email search
            user = mcp_server.get_user_by_email(search_term)
            if user:
                print(f"✅ Found: {user.first_name} {user.last_name} ({user.email})")
            else:
                print(f"❌ Not found: {search_term}")
    
    return status['status'] == 'connected'


def main():
    """Run all tests"""
    print("🧪 Tax Chatbot Router Test Suite")
    print("=" * 60)
    
    # Run tests
    routing_success = test_routing()
    database_success = test_database()
    
    # Overall result
    print("\n" + "=" * 60)
    if routing_success and database_success:
        print("🎉 All tests passed! Router-based tax chatbot is ready.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        
    print("\n💡 To test the full system, run:")
    print("   python src/interfaces/cli_chat.py")
    print("\n📝 Example queries to try:")
    print("   • What is B&O tax? (general)")
    print("   • What are Sarah Johnson's tax obligations? (personal)")
    print("   • Show me Michael Chen's filing status (personal)")


if __name__ == "__main__":
    main()