# Tax Chatbot Examples

This directory contains example scripts and usage patterns for the Tax Chatbot system.

## Quick Start Examples

### 1. Basic Query Examples (`basic_queries.py`)
Simple examples showing different types of tax queries and expected responses.

### 2. Personal Tax Queries (`personal_queries.py`)
Examples using the demo users to show personalized tax assistance.

### 3. Jurisdiction-Specific Queries (`jurisdiction_examples.py`)
Examples demonstrating California vs general tax query handling.

### 4. System Testing (`system_tests.py`)
Examples for testing system components and performance.

## Running Examples

```bash
# Make sure you're in the project root
cd tax-chatbot

# Run basic query examples
python examples/basic_queries.py

# Test personal queries with demo users
python examples/personal_queries.py

# Test jurisdiction detection
python examples/jurisdiction_examples.py
```

## Demo Users

The system includes three fictional users for testing:

| Name | Email | Tax ID | Filing Status |
|------|-------|---------|---------------|
| Sarah Johnson | sarah.johnson@email.com | 123-45-6789 | Single |
| Michael Chen | michael.chen@email.com | 987-65-4321 | Married |
| Emily Rodriguez | emily.rodriguez@email.com | 456-78-9123 | Single |

## Expected Output

All examples include expected output comments to help you verify the system is working correctly.