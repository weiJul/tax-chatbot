#!/usr/bin/env python3
import sqlite3
import os
from datetime import datetime

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, 'customers.db')

# Create database and table
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create customers table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    tax_id TEXT UNIQUE NOT NULL,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    zip_code TEXT NOT NULL,
    filing_status TEXT NOT NULL CHECK (filing_status IN ('single', 'married_jointly', 'married_separately', 'head_of_household', 'qualifying_widow')),
    annual_income DECIMAL(12,2) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

# Insert sample customer data
sample_customers = [
    {
        'first_name': 'Sarah',
        'last_name': 'Johnson',
        'email': 'sarah.johnson@email.com',
        'phone': '555-0123',
        'tax_id': '123-45-6789',
        'address': '123 Main Street',
        'city': 'Austin',
        'state': 'TX',
        'zip_code': '78701',
        'filing_status': 'single',
        'annual_income': 75000.00
    },
    {
        'first_name': 'Michael',
        'last_name': 'Chen',
        'email': 'michael.chen@email.com',
        'phone': '555-0456',
        'tax_id': '234-56-7890',
        'address': '456 Oak Avenue',
        'city': 'San Francisco',
        'state': 'CA',
        'zip_code': '94102',
        'filing_status': 'married_jointly',
        'annual_income': 125000.00
    },
    {
        'first_name': 'Emily',
        'last_name': 'Rodriguez',
        'email': 'emily.rodriguez@email.com',
        'phone': '555-0789',
        'tax_id': '345-67-8901',
        'address': '789 Pine Street',
        'city': 'Denver',
        'state': 'CO',
        'zip_code': '80202',
        'filing_status': 'head_of_household',
        'annual_income': 92000.00
    }
]

# Insert sample data
for customer in sample_customers:
    cursor.execute('''
        INSERT OR IGNORE INTO customers 
        (first_name, last_name, email, phone, tax_id, address, city, state, zip_code, filing_status, annual_income)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        customer['first_name'], customer['last_name'], customer['email'], customer['phone'],
        customer['tax_id'], customer['address'], customer['city'], customer['state'],
        customer['zip_code'], customer['filing_status'], customer['annual_income']
    ))

conn.commit()

# Verify data was inserted
cursor.execute('SELECT COUNT(*) FROM customers')
count = cursor.fetchone()[0]
print(f"Database created successfully with {count} customers.")

# Show sample of data
cursor.execute('SELECT first_name, last_name, email, filing_status, annual_income FROM customers')
customers = cursor.fetchall()
print("\nSample customer data:")
for customer in customers:
    print(f"- {customer[0]} {customer[1]} ({customer[2]}) - {customer[3]}, ${customer[4]:,.2f}")

conn.close()
print(f"\nDatabase saved to: {db_path}")