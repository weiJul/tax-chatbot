-- Tax App Customer Database Setup Script
-- This script creates a simple customer database for a tax application
-- Designed for MCP/LLM integration for customer support queries

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    phone TEXT,
    tax_id TEXT UNIQUE NOT NULL,           -- Social Security Number or Tax ID
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    zip_code TEXT NOT NULL,
    filing_status TEXT NOT NULL CHECK (filing_status IN (
        'single', 
        'married_jointly', 
        'married_separately', 
        'head_of_household', 
        'qualifying_widow'
    )),
    annual_income DECIMAL(12,2) NOT NULL,  -- Annual gross income for tax calculations
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample customer data for testing
INSERT OR IGNORE INTO customers 
(first_name, last_name, email, phone, tax_id, address, city, state, zip_code, filing_status, annual_income)
VALUES 
-- Customer 1: Single filer, moderate income
('Sarah', 'Johnson', 'sarah.johnson@email.com', '555-0123', '123-45-6789', 
 '123 Main Street', 'Austin', 'TX', '78701', 'single', 75000.00),

-- Customer 2: Married filing jointly, higher income
('Michael', 'Chen', 'michael.chen@email.com', '555-0456', '234-56-7890', 
 '456 Oak Avenue', 'San Francisco', 'CA', '94102', 'married_jointly', 125000.00),

-- Customer 3: Head of household, supporting dependents
('Emily', 'Rodriguez', 'emily.rodriguez@email.com', '555-0789', '345-67-8901', 
 '789 Pine Street', 'Denver', 'CO', '80202', 'head_of_household', 92000.00);

-- Create useful indexes for MCP queries
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_tax_id ON customers(tax_id);
CREATE INDEX IF NOT EXISTS idx_customers_filing_status ON customers(filing_status);
CREATE INDEX IF NOT EXISTS idx_customers_state ON customers(state);

-- Sample queries for MCP/LLM integration:
-- SELECT * FROM customers WHERE filing_status = 'single';
-- SELECT first_name, last_name, annual_income FROM customers WHERE annual_income > 100000;
-- SELECT * FROM customers WHERE state = 'CA';
-- SELECT AVG(annual_income) FROM customers GROUP BY filing_status;