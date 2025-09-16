"""
Query Router for Tax Chatbot
Uses LangChain to classify queries as either personal user data requests or general tax information
"""
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import Runnable

from .config import config_manager
from .tax_mcp_client import mcp_server  # Use pure MCP client
from .tax_mcp_server import UserData  # Import UserData from MCP server

logger = logging.getLogger(__name__)


@dataclass
class QueryClassification:
    """Query classification result"""
    query_type: str  # "user_data" or "general_tax"
    confidence: float
    user_identifier: Optional[str] = None
    reasoning: Optional[str] = None


@dataclass
class UserContext:
    """User context for personalized responses"""
    user_data: Optional[UserData] = None
    tax_context: Optional[Dict[str, Any]] = None
    found: bool = False


class QueryTypeParser(BaseOutputParser[QueryClassification]):
    """Parser for query classification output"""
    
    def parse(self, text: str) -> QueryClassification:
        """Parse LLM output into QueryClassification"""
        try:
            # Extract classification from LLM response
            if "user_data" in text.lower():
                query_type = "user_data"
            elif "general_tax" in text.lower():
                query_type = "general_tax"
            else:
                # Default fallback
                query_type = "general_tax"
            
            # Extract confidence score (simple heuristic)
            confidence = 0.8  # Default confidence
            if "definitely" in text.lower() or "clearly" in text.lower():
                confidence = 0.9
            elif "might" in text.lower() or "possibly" in text.lower():
                confidence = 0.6
            
            # Extract user identifier if mentioned
            user_identifier = self._extract_user_identifier(text)
            
            return QueryClassification(
                query_type=query_type,
                confidence=confidence,
                user_identifier=user_identifier,
                reasoning=text.strip()
            )
            
        except Exception as e:
            logger.error(f"Failed to parse query classification: {str(e)}")
            # Return safe default
            return QueryClassification(
                query_type="general_tax",
                confidence=0.5,
                reasoning="Parse error, defaulting to general"
            )
    
    def _extract_user_identifier(self, text: str) -> Optional[str]:
        """Extract user identifier from LLM response"""
        # Look for names, emails, or tax IDs mentioned
        patterns = [
            r"name:\s*([A-Za-z\s]+)",
            r"user:\s*([A-Za-z\s]+)",
            r"email:\s*([A-Za-z0-9@.]+)",
            r"tax[_\s]id:\s*([0-9-]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


class QueryRouter:
    """
    Query router that classifies user queries and routes them appropriately
    Uses LangChain for intelligent classification
    """
    
    def __init__(self, llm: Optional[Runnable] = None):
        """Initialize query router"""
        self.config = config_manager.rag
        self.llm = llm  # Will be injected from LLM service
        
        # Classification prompt template
        self.classification_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are a tax chatbot query classifier. Analyze the user query and classify it as either:

1. "user_data" - Query requires personal tax information from database (MUST mention specific names, emails, tax IDs, or possessive pronouns)
2. "general_tax" - Query about general tax rules, procedures, deadlines, or information (includes questions about "when", "how", "what" without personal identifiers)

Examples of user_data queries (REQUIRE SPECIFIC PERSONAL IDENTIFIERS):
- "What are Sarah Johnson's tax obligations?"
- "Show me John Smith's filing status" 
- "What is michael.chen@email.com's tax bracket?"
- "Calculate taxes for SSN 123-45-6789"
- "What are MY tax obligations?" (possessive)
- "Show me MY filing status" (possessive)

Examples of general_tax queries (NO PERSONAL IDENTIFIERS):
- "What is the B&O tax rate?"
- "When are tax deadlines?"
- "When do I need to file taxes?" (general question)
- "When do I need to return taxes?" (general question)
- "How do I file taxes in Washington State?"
- "What are the tax brackets for 2023?"
- "When are taxes due?"
- "How much is the penalty for late filing?"

IMPORTANT: Only classify as "user_data" if the query contains:
- Specific person names (Sarah Johnson, John Smith)
- Email addresses (user@email.com)
- Tax ID numbers (123-45-6789)
- Possessive pronouns referring to personal data (MY taxes, MY status)

Questions like "When do I need to..." or "How do I..." are GENERAL unless they reference specific personal data.

Query to classify: "{query}"

Classification: [user_data or general_tax]
Confidence: [high/medium/low]
User identifier (if any): [name, email, or tax ID mentioned]
Reasoning: [brief explanation]
"""
        )
        
        # User identification prompt for extracting user info from queries
        self.user_extraction_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
Extract user identification information from this query:

Query: "{query}"

Look for:
- Full names (first and last name)
- Email addresses 
- Tax ID numbers (SSN format like 123-45-6789)

If found, return in this format:
Type: [name/email/tax_id]
Value: [the actual value]

If no user identifier found, return: "None"
"""
        )
        
        logger.info("Query router initialized")
    
    def set_llm(self, llm: Runnable) -> None:
        """Set the LLM for the router"""
        self.llm = llm
        logger.info("LLM set for query router")
    
    def classify_query(self, query: str) -> QueryClassification:
        """Classify query type and extract user information"""
        try:
            # Fallback classification if no LLM available
            if self.llm is None:
                return self._fallback_classify(query)
            
            # Use modern Runnable syntax instead of deprecated LLMChain
            parser = QueryTypeParser()
            chain = self.classification_prompt | self.llm | parser
            
            result = chain.invoke({"query": query})
            logger.info(f"Query classified as: {result.query_type} (confidence: {result.confidence})")
            
            return result
            
        except Exception as e:
            logger.error(f"Classification failed, using fallback: {str(e)}")
            return self._fallback_classify(query)
    
    def extract_user_identifier(self, query: str) -> Optional[str]:
        """Extract user identifier from query using multiple methods"""
        try:
            # Method 1: Enhanced pattern matching (primary method)
            pattern_result = self._pattern_extract_user(query)
            if pattern_result:
                logger.info(f"Pattern extraction found: {pattern_result}")
                return pattern_result
            
            # Method 2: LLM extraction (backup method)
            if self.llm is not None:
                chain = self.user_extraction_prompt | self.llm
                result = chain.invoke({"query": query})
                
                # Parse LLM response
                if "None" not in result and "Value:" in result:
                    lines = result.strip().split('\n')
                    for line in lines:
                        if line.startswith("Value:"):
                            extracted = line.split(":", 1)[1].strip()
                            logger.info(f"LLM extraction found: {extracted}")
                            return extracted
            
            logger.info("No user identifier found in query")
            return None
            
        except Exception as e:
            logger.error(f"User extraction failed: {str(e)}")
            # Fallback to pattern matching
            return self._pattern_extract_user(query)
    
    def find_user_data(self, user_identifier: str) -> UserContext:
        """Find user data using MCP server with smart identifier handling"""
        try:
            user_data = None
            search_method = "unknown"
            
            # Clean the identifier
            clean_identifier = user_identifier.strip()
            
            # Try different search methods based on identifier format
            if "@" in clean_identifier:
                # Email format
                search_method = "email"
                user_data = mcp_server.get_user_by_email(clean_identifier)
                logger.info(f"Searching by email: {clean_identifier}")
                
            elif re.match(r'\d{3}-\d{2}-\d{4}', clean_identifier):
                # Tax ID format (SSN)
                search_method = "tax_id"
                user_data = mcp_server.get_user_by_tax_id(clean_identifier)
                logger.info(f"Searching by tax ID: {clean_identifier}")
                
            else:
                # Assume name format - try multiple parsing strategies
                search_method = "name"
                user_data = self._search_by_name_variations(clean_identifier)
            
            if user_data:
                # Try to get tax context via MCP, fallback to local calculation
                try:
                    tax_context = mcp_server.get_user_tax_data(user_data.id)
                    # If MCP fails or returns empty, calculate locally
                    if not tax_context:
                        tax_context = self._calculate_local_tax_context(user_data)
                except Exception as e:
                    logger.warning(f"MCP tax context failed, using local calculation: {str(e)}")
                    tax_context = self._calculate_local_tax_context(user_data)
                
                logger.info(f"Successfully found user: {user_data.first_name} {user_data.last_name} (method: {search_method})")
                return UserContext(
                    user_data=user_data,
                    tax_context=tax_context,
                    found=True
                )
            else:
                logger.info(f"No user found for identifier: {user_identifier} (method: {search_method})")
                return UserContext(found=False)
                
        except Exception as e:
            logger.error(f"Failed to find user data for {user_identifier}: {str(e)}")
            return UserContext(found=False)
    
    def _search_by_name_variations(self, name_identifier: str) -> Optional[UserData]:
        """Search for user by trying different name parsing strategies"""
        
        # Strategy 1: Split by spaces (standard "First Last")
        name_parts = name_identifier.strip().split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = " ".join(name_parts[1:])  # Handle multiple last names
            
            logger.info(f"Trying name search: '{first_name}' '{last_name}'")
            user_data = mcp_server.get_user_by_name(first_name, last_name)
            if user_data:
                return user_data
        
        # Strategy 2: Try reversed order ("Last, First" or "Last First")
        if len(name_parts) == 2:
            logger.info(f"Trying reversed name search: '{name_parts[1]}' '{name_parts[0]}'")
            user_data = mcp_server.get_user_by_name(name_parts[1], name_parts[0])
            if user_data:
                return user_data
        
        # Strategy 3: Handle comma-separated names ("Last, First")
        if "," in name_identifier:
            parts = name_identifier.split(",")
            if len(parts) == 2:
                last_name = parts[0].strip()
                first_name = parts[1].strip()
                logger.info(f"Trying comma-separated name search: '{first_name}' '{last_name}'")
                user_data = mcp_server.get_user_by_name(first_name, last_name)
                if user_data:
                    return user_data
        
        # Strategy 4: General search (search in all fields)
        logger.info(f"Trying general search for: {name_identifier}")
        users = mcp_server.search_users(name_identifier)
        if users:
            return users[0]  # Return the first match
        
        return None
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Intelligent sequential routing with name-based context enhancement
        
        Flow:
        1. Name Detection - Check if query contains person's name
        2. Database Lookup - If name found, search database via MCP  
        3. Context Enhancement - If database success, add user info as context
        4. Universal Fallback - Always proceed to general search with optional enhanced context
        
        Returns:
            Dictionary with routing decision and enhanced context
        """
        logger.debug(f"Starting intelligent sequential routing for: '{query}'")
        
        result = {
            'original_query': query,
            'query_type': 'general_search',  # Always general search as final step
            'name_detected': False,
            'database_lookup_attempted': False,
            'database_lookup_success': False,
            'user_context': None,
            'context_enhanced': False,
            'extraction_attempts': [],
            'routing_path': []
        }
        
        # Step 1: Name Detection - Check if query contains person's name
        logger.debug("Step 1: Name Detection")
        result['routing_path'].append('name_detection')
        
        user_identifiers = self._extract_all_user_identifiers(query)
        result['extraction_attempts'] = user_identifiers
        
        if user_identifiers:
            result['name_detected'] = True
            logger.info(f"Name detected in query: {[info['identifier'] for info in user_identifiers[:2]]}")
            
            # Step 2: Database Lookup - Search database via MCP for user information
            logger.debug("Step 2: Database Lookup")
            result['routing_path'].append('database_lookup')
            result['database_lookup_attempted'] = True
            
            # Try each identifier until we find a user
            for identifier_info in user_identifiers:
                identifier = identifier_info['identifier']
                method = identifier_info['method']
                confidence = identifier_info['confidence']
                
                logger.debug(f"Attempting database lookup for '{identifier}' (method: {method})")
                
                user_context = self.find_user_data(identifier)
                if user_context.found:
                    result['database_lookup_success'] = True
                    result['user_context'] = user_context
                    result['user_identifier'] = identifier
                    result['successful_method'] = method
                    logger.info(f"Database lookup successful for: {identifier}")
                    
                    # Step 3: Context Enhancement - Add retrieved user info as context
                    logger.debug("Step 3: Context Enhancement")
                    result['routing_path'].append('context_enhancement')
                    result['context_enhanced'] = True
                    logger.info("User context will be added to general search")
                    break  # Stop on first successful lookup
            
            if not result['database_lookup_success']:
                attempted_ids = [info['identifier'] for info in user_identifiers[:3]]
                logger.info(f"Database lookup failed for all identifiers: {attempted_ids}")
        else:
            logger.debug("No names detected in query")
        
        # Step 4: Universal Fallback - Always proceed to general search
        logger.debug("Step 4: Universal Fallback - Proceeding to general search")
        result['routing_path'].append('general_search')
        
        # Determine the enhanced query type based on context availability
        if result['context_enhanced']:
            result['query_type'] = 'general_search_with_context'
            logger.info("Route: General search with enhanced user context")
        else:
            result['query_type'] = 'general_search'
            logger.info("Route: Standard general search")
        
        # Add summary information
        result['routing_summary'] = self._create_routing_summary(result)
        
        logger.debug(f"Sequential routing complete: {' -> '.join(result['routing_path'])}")
        return result
    
    def _extract_all_user_identifiers(self, query: str) -> List[Dict[str, Any]]:
        """Extract user identifiers with early exit optimization"""
        identifiers = []
        
        # Method 1: Enhanced pattern matching (fastest, most reliable)
        try:
            pattern_result = self._pattern_extract_user(query)
            if pattern_result:
                identifiers.append({
                    'identifier': pattern_result,
                    'method': 'pattern_matching',
                    'confidence': 0.8
                })
                # If we got a good pattern match, try it first before doing more expensive operations
                logger.debug(f"Found pattern match: {pattern_result}")
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
        
        # Method 2: Simple patterns as backup (fast)
        if not identifiers:  # Only if pattern matching didn't find anything
            try:
                simple_patterns = self._simple_pattern_extract(query)
                for pattern in simple_patterns:
                    identifiers.append({
                        'identifier': pattern,
                        'method': 'simple_patterns',
                        'confidence': 0.6
                    })
                    break  # Take first simple pattern and try it
            except Exception as e:
                logger.error(f"Simple pattern extraction failed: {e}")
        
        # Method 3: LLM extraction (slower, only if needed)
        if not identifiers and self.llm is not None:
            try:
                llm_result = self._llm_extract_user(query)
                if llm_result:
                    identifiers.append({
                        'identifier': llm_result,
                        'method': 'llm_extraction',
                        'confidence': 0.7
                    })
            except Exception as e:
                logger.error(f"LLM extraction failed: {e}")
        
        # Sort by confidence and return top 2 to limit database queries
        identifiers.sort(key=lambda x: x['confidence'], reverse=True)
        return identifiers[:2]  # Limit to top 2 candidates
    
    def _llm_extract_user(self, query: str) -> Optional[str]:
        """Extract user identifier using LLM"""
        if self.llm is None:
            return None
        
        chain = self.user_extraction_prompt | self.llm
        result = chain.invoke({"query": query})
        
        # Parse LLM response
        if "None" not in result and "Value:" in result:
            lines = result.strip().split('\n')
            for line in lines:
                if line.startswith("Value:"):
                    extracted = line.split(":", 1)[1].strip()
                    return extracted
        
        return None
    
    def _simple_pattern_extract(self, query: str) -> List[str]:
        """Simple fallback patterns for user identification"""
        results = []
        
        # Simple name pattern (any capitalized word pairs)
        simple_names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', query)
        for name in simple_names:
            if self._is_valid_name(name):
                results.append(name)
        
        # Email pattern
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
        results.extend(emails)
        
        # Tax ID pattern
        tax_ids = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', query)
        results.extend(tax_ids)
        
        return results
    
    def _fallback_classify(self, query: str) -> QueryClassification:
        """Fallback classification using simple pattern matching"""
        query_lower = query.lower()
        
        # Personal indicators (require possessive or specific personal context)
        personal_indicators = [
            "my tax", "my filing", "my income", "my return",
            "i owe", "i need to pay", "my status", "my bracket",
            "show me my", "find my"
        ]
        
        # Name patterns
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Tax ID pattern
        tax_id_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        
        # Check for personal indicators
        has_personal = any(indicator in query_lower for indicator in personal_indicators)
        has_name = bool(re.search(name_pattern, query))
        has_email = bool(re.search(email_pattern, query))
        has_tax_id = bool(re.search(tax_id_pattern, query))
        
        if has_personal or has_name or has_email or has_tax_id:
            return QueryClassification(
                query_type="user_data",
                confidence=0.7,
                reasoning="Fallback: Contains personal indicators or identifiers"
            )
        else:
            return QueryClassification(
                query_type="general_tax",
                confidence=0.8,
                reasoning="Fallback: No personal indicators found"
            )
    
    def _pattern_extract_user(self, query: str) -> Optional[str]:
        """Extract user identifier using enhanced regex patterns"""
        candidates = []
        
        # Email pattern (highest priority)
        email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query)
        for email in email_matches:
            candidates.append(("email", email, 1.0))
        
        # Tax ID pattern (high priority)
        tax_id_matches = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', query)
        for tax_id in tax_id_matches:
            candidates.append(("tax_id", tax_id, 0.9))
        
        # Enhanced name extraction patterns
        name_patterns = [
            # "My name is John Doe" - most common conversational pattern
            (r'(?:my\s+name\s+is\s+|i\s+am\s+|i\'m\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 0.8),
            # "John Doe here" or "This is John Doe"
            (r'(?:this\s+is\s+|here\s+is\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', 0.7),
            # Direct name mentions with context words
            (r'(?:for\s+|about\s+|regarding\s+)([A-Z][a-z]+\s+[A-Z][a-z]+)', 0.6),
            # Names in possessive form "John's taxes"
            (r'([A-Z][a-z]+\s+[A-Z][a-z]+)\'s\s+', 0.6),
            # Basic pattern: "First Last" (capitalized words)
            (r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 0.5),
        ]
        
        for pattern, confidence in name_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Clean up the match and validate
                clean_name = self._clean_extracted_name(match)
                if clean_name and self._is_valid_name(clean_name):
                    candidates.append(("name", clean_name, confidence))
        
        # Sort by confidence (highest first) and return the best candidate
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            logger.debug(f"Name extraction candidates: {[(c[1], c[2]) for c in candidates[:3]]}")
            return candidates[0][1]  # Return the highest confidence match
        
        return None
    
    def _clean_extracted_name(self, name_match: str) -> str:
        """Clean and normalize extracted name"""
        if not name_match:
            return ""
        
        # Remove extra whitespace and normalize
        clean_name = ' '.join(name_match.split())
        
        # Title case the name properly
        clean_name = ' '.join(word.capitalize() for word in clean_name.split())
        
        return clean_name
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate that the extracted string looks like a real name"""
        if not name or len(name.strip()) < 3:
            return False
        
        # Split into words
        words = name.split()
        
        # Must have at least 2 words (first + last name)
        if len(words) < 2:
            return False
        
        # Check for invalid patterns
        invalid_patterns = [
            # Common phrases that aren't names
            r'\b(my\s+name|the\s+user|washington\s+state|tax\s+id|social\s+security)\b',
            # Query words that aren't names
            r'\b(when\s+do|how\s+do|what\s+do|where\s+do|why\s+do|can\s+i|should\s+i)\b',
            # Names ending with conjunctions or common words
            r'\b\w+\s+\w+\s+(and|or|the|is|are|have|has|with|for|about|that|which|wonder|think|need|want)\b',
            # All caps (likely not a name)
            r'^[A-Z\s]+$',
            # Contains numbers
            r'\d',
            # Too many words (likely a sentence fragment)
            r'^\S+\s+\S+\s+\S+\s+\S+\s+\S+'  # More than 4 words
        ]
        
        name_lower = name.lower()
        for pattern in invalid_patterns:
            if re.search(pattern, name_lower):
                logger.debug(f"Rejected name candidate '{name}' - matches invalid pattern: {pattern}")
                return False
        
        # Each word should look like a name (start with letter, mostly letters)
        for word in words:
            if not re.match(r'^[A-Za-z][A-Za-z\'\-]{1,}$', word):
                logger.debug(f"Rejected name candidate '{name}' - invalid word: {word}")
                return False
        
        logger.debug(f"Validated name candidate: '{name}'")
        return True
    
    def _create_routing_summary(self, result: Dict[str, Any]) -> str:
        """Create a human-readable summary of the routing process"""
        summary_parts = []
        
        if result['name_detected']:
            names = [info['identifier'] for info in result['extraction_attempts'][:2]]
            summary_parts.append(f"Names detected: {', '.join(names)}")
        else:
            summary_parts.append("No names detected")
        
        if result['database_lookup_attempted']:
            if result['database_lookup_success']:
                summary_parts.append(f"Database lookup successful ({result.get('successful_method', 'unknown')})")
            else:
                summary_parts.append("Database lookup failed")
        
        if result['context_enhanced']:
            user = result['user_context'].user_data
            summary_parts.append(f"Context enhanced with {user.first_name} {user.last_name}'s data")
        
        summary_parts.append(f"Final route: {result['query_type']}")
        
        return " | ".join(summary_parts)
    
    def _calculate_local_tax_context(self, user_data) -> Dict[str, Any]:
        """Calculate tax context locally when MCP fails"""
        try:
            # Calculate tax bracket based on 2023 rates (single filer)
            income = float(user_data.annual_income)
            if income <= 11000:
                tax_bracket = "10%"
            elif income <= 44725:
                tax_bracket = "12%"
            elif income <= 95375:
                tax_bracket = "22%"
            elif income <= 182050:
                tax_bracket = "24%"
            elif income <= 231250:
                tax_bracket = "32%"
            elif income <= 578125:
                tax_bracket = "35%"
            else:
                tax_bracket = "37%"
            
            return {
                "user_id": user_data.id,
                "name": f"{user_data.first_name} {user_data.last_name}",
                "filing_status": user_data.filing_status,
                "annual_income": float(user_data.annual_income),
                "tax_bracket": tax_bracket,
                "state": user_data.state
            }
        except Exception as e:
            logger.error(f"Local tax context calculation failed: {str(e)}")
            return {}


# Global query router instance
query_router = QueryRouter()