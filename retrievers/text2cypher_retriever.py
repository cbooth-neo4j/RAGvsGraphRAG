"""
Text2Cypher Retriever - Natural Language to Cypher Query Translation

This module implements Text2Cypher using Neo4j with natural language
to Cypher query translation for direct graph database querying.

Features:
- Iterative refinement with verification and correction loops
- Configurable Text2Cypher-specific model (TEXT2CYPHER_MODEL)
- Multiple verification methods (syntax, execution, LLM)
- Rule-based and LLM-based correction

All verification and correction code is inlined in this module.
"""

import os
import re
import json
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import neo4j
from neo4j_graphrag.retrievers import Text2CypherRetriever
from langchain_neo4j import Neo4jGraph

# Import centralized configuration
from config import get_model_config, get_neo4j_embeddings, get_text2cypher_llm, get_text2cypher_langchain_llm, ModelProvider

from utils.graph_rag_logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


# =============================================================================
# VERIFICATION CLASSES
# =============================================================================

class VerifierType(Enum):
    """Types of verification methods available"""
    SYNTAX = "syntax"
    EXECUTION = "execution"
    LLM = "llm"


@dataclass
class VerificationResult:
    """Result of a Cypher query verification"""
    is_valid: bool
    verifier_type: VerifierType
    error_message: Optional[str] = None
    error_details: Optional[dict] = None
    
    def __str__(self) -> str:
        if self.is_valid:
            return f"[{self.verifier_type.value}] Valid"
        return f"[{self.verifier_type.value}] Invalid: {self.error_message}"


class CypherVerifier(ABC):
    """Abstract base class for Cypher query verifiers"""
    
    @property
    @abstractmethod
    def verifier_type(self) -> VerifierType:
        """Return the type of this verifier"""
        pass
    
    @abstractmethod
    def verify(self, cypher_query: str, schema: str = None, **kwargs) -> VerificationResult:
        """Verify a Cypher query"""
        pass


class SyntaxVerifier(CypherVerifier):
    """Rule-based Cypher syntax verification - fast validation using regex patterns"""
    
    @property
    def verifier_type(self) -> VerifierType:
        return VerifierType.SYNTAX
    
    VALID_KEYWORDS = {
        'MATCH', 'OPTIONAL MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 
        'DELETE', 'SET', 'REMOVE', 'MERGE', 'CALL', 'UNWIND', 'FOREACH',
        'ORDER BY', 'SKIP', 'LIMIT', 'UNION', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
    }
    
    ERROR_PATTERNS = [
        (r'^\s*$', "Empty query"),
        (r'MATCH\s*\(\s*\)', "Empty node pattern - missing label or variable"),
        (r'\[\s*:\s*\]', "Empty relationship type"),
        (r'RETURN\s*$', "RETURN clause without any expressions"),
        (r'WHERE\s*$', "WHERE clause without conditions"),
        (r'\(\s*\)', "Empty parentheses - incomplete pattern"),
        (r'\[\s*\]', "Empty brackets - incomplete relationship"),
        (r'[^\w]=[^\w=]', "Invalid assignment syntax"),
    ]
    
    def verify(self, cypher_query: str, schema: str = None, **kwargs) -> VerificationResult:
        if not cypher_query or not cypher_query.strip():
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message="Empty or whitespace-only query"
            )
        
        query_upper = cypher_query.upper()
        
        has_valid_start = any(
            keyword in query_upper 
            for keyword in ['MATCH', 'CREATE', 'MERGE', 'CALL', 'UNWIND', 'RETURN', 'WITH']
        )
        
        if not has_valid_start:
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message="Query doesn't start with a valid Cypher clause",
                error_details={'query_preview': cypher_query[:100]}
            )
        
        for pattern, error_msg in self.ERROR_PATTERNS:
            if re.search(pattern, cypher_query, re.IGNORECASE):
                return VerificationResult(
                    is_valid=False,
                    verifier_type=self.verifier_type,
                    error_message=error_msg,
                    error_details={'pattern': pattern}
                )
        
        balance_result = self._check_balanced_delimiters(cypher_query)
        if not balance_result[0]:
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message=balance_result[1]
            )
        
        if 'MATCH' in query_upper and 'RETURN' not in query_upper:
            if 'CALL' not in query_upper and 'CREATE' not in query_upper:
                return VerificationResult(
                    is_valid=False,
                    verifier_type=self.verifier_type,
                    error_message="MATCH clause without RETURN - query won't return results"
                )
        
        logger.debug(f"Syntax verification passed for query: {cypher_query[:50]}...")
        return VerificationResult(is_valid=True, verifier_type=self.verifier_type)
    
    def _check_balanced_delimiters(self, query: str) -> Tuple[bool, str]:
        """Check that parentheses, brackets, and braces are balanced"""
        stack = []
        matching = {')': '(', ']': '[', '}': '{'}
        in_string = False
        string_char = None
        
        for i, char in enumerate(query):
            if char in ('"', "'") and (i == 0 or query[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue
            
            if in_string:
                continue
                
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack:
                    return False, f"Unmatched closing '{char}' at position {i}"
                if stack[-1] != matching[char]:
                    return False, f"Mismatched delimiter: expected '{matching[char]}' but found '{char}'"
                stack.pop()
        
        if stack:
            return False, f"Unclosed delimiter(s): {stack}"
        return True, ""


class ExecutionVerifier(CypherVerifier):
    """Execution-based verification using EXPLAIN - validates against actual database schema"""
    
    @property
    def verifier_type(self) -> VerifierType:
        return VerifierType.EXECUTION
    
    def __init__(self, driver=None):
        self.driver = driver
    
    def set_driver(self, driver):
        self.driver = driver
    
    def verify(self, cypher_query: str, schema: str = None, **kwargs) -> VerificationResult:
        driver = kwargs.get('driver', self.driver)
        database = kwargs.get('database')
        
        if driver is None:
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message="No Neo4j driver available for execution verification"
            )
        
        query_upper = cypher_query.upper().strip()
        if query_upper.startswith('CALL ') and 'YIELD' not in query_upper:
            logger.debug("Skipping EXPLAIN for procedure call without YIELD")
            return VerificationResult(
                is_valid=True,
                verifier_type=self.verifier_type,
                error_details={'skipped': 'procedure_call'}
            )
        
        try:
            explain_query = f"EXPLAIN {cypher_query}"
            with driver.session(database=database) as session:
                result = session.run(explain_query)
                result.consume()
            
            logger.debug(f"Execution verification passed for query: {cypher_query[:50]}...")
            return VerificationResult(is_valid=True, verifier_type=self.verifier_type)
            
        except Exception as e:
            error_msg = str(e)
            error_details = {'raw_error': error_msg}
            
            if 'SyntaxError' in error_msg or 'Invalid input' in error_msg:
                error_details['error_type'] = 'syntax'
            elif 'Unknown' in error_msg and ('label' in error_msg.lower() or 'type' in error_msg.lower()):
                error_details['error_type'] = 'schema_mismatch'
            elif 'variable' in error_msg.lower() and 'not defined' in error_msg.lower():
                error_details['error_type'] = 'undefined_variable'
            
            logger.debug(f"Execution verification failed: {error_msg[:100]}")
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message=error_msg[:500],
                error_details=error_details
            )


class LLMVerifier(CypherVerifier):
    """LLM-based semantic verification - most thorough but slowest"""
    
    @property
    def verifier_type(self) -> VerifierType:
        return VerifierType.LLM
    
    VERIFICATION_PROMPT = """You are a Cypher query validator. Analyze the following Cypher query and determine if it is valid.

## Database Schema
{schema}

## Cypher Query to Validate
```cypher
{query}
```

## Validation Tasks
1. Check if node labels exist in the schema
2. Check if relationship types exist in the schema
3. Check if property names are valid
4. Check if the query logic makes sense for the schema
5. Check for relationship direction correctness

## Response Format
Respond with ONLY a JSON object (no markdown, no explanation):
{{"is_valid": true/false, "issues": ["issue1", "issue2"] or []}}"""

    def __init__(self, llm=None):
        self.llm = llm
    
    def set_llm(self, llm):
        self.llm = llm
    
    def verify(self, cypher_query: str, schema: str = None, **kwargs) -> VerificationResult:
        llm = kwargs.get('llm', self.llm)
        
        if llm is None:
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message="No LLM available for semantic verification"
            )
        
        if not schema:
            logger.warning("LLM verification without schema - limited validation possible")
            schema = "Schema not available - validate syntax only"
        
        try:
            prompt = self.VERIFICATION_PROMPT.format(schema=schema, query=cypher_query)
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            response_text = response_text.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            try:
                result = json.loads(response_text)
                is_valid = result.get('is_valid', False)
                issues = result.get('issues', [])
                
                if is_valid:
                    logger.debug(f"LLM verification passed for query: {cypher_query[:50]}...")
                    return VerificationResult(is_valid=True, verifier_type=self.verifier_type)
                else:
                    return VerificationResult(
                        is_valid=False,
                        verifier_type=self.verifier_type,
                        error_message="; ".join(issues) if issues else "LLM flagged query as invalid",
                        error_details={'issues': issues}
                    )
            except json.JSONDecodeError:
                response_lower = response_text.lower()
                if 'valid' in response_lower and 'invalid' not in response_lower:
                    return VerificationResult(
                        is_valid=True,
                        verifier_type=self.verifier_type,
                        error_details={'raw_response': response_text[:200]}
                    )
                else:
                    return VerificationResult(
                        is_valid=False,
                        verifier_type=self.verifier_type,
                        error_message="Could not parse LLM validation response",
                        error_details={'raw_response': response_text[:500]}
                    )
                    
        except Exception as e:
            logger.error(f"LLM verification error: {e}")
            return VerificationResult(
                is_valid=False,
                verifier_type=self.verifier_type,
                error_message=f"LLM verification failed: {str(e)}"
            )


class CypherVerificationPipeline:
    """Pipeline for running multiple verifiers in order (fast to slow, stops on first failure)"""
    
    def __init__(self, verifiers: List[CypherVerifier] = None, stop_on_first_failure: bool = True):
        self.verifiers = verifiers or [SyntaxVerifier()]
        self.stop_on_first_failure = stop_on_first_failure
    
    def add_verifier(self, verifier: CypherVerifier):
        self.verifiers.append(verifier)
    
    def verify(self, cypher_query: str, schema: str = None, **kwargs) -> Tuple[bool, List[VerificationResult]]:
        results = []
        all_passed = True
        
        for verifier in self.verifiers:
            result = verifier.verify(cypher_query, schema, **kwargs)
            results.append(result)
            
            if not result.is_valid:
                all_passed = False
                logger.debug(f"Verification failed at {verifier.verifier_type.value}: {result.error_message}")
                if self.stop_on_first_failure:
                    break
        
        return all_passed, results
    
    def get_error_summary(self, results: List[VerificationResult]) -> str:
        errors = [
            f"[{r.verifier_type.value}] {r.error_message}"
            for r in results
            if not r.is_valid and r.error_message
        ]
        return "\n".join(errors) if errors else "No errors"


def create_verification_pipeline(
    enable_syntax: bool = True,
    enable_execution: bool = True,
    enable_llm: bool = False,
    driver=None,
    llm=None
) -> CypherVerificationPipeline:
    """Factory function to create a verification pipeline"""
    verifiers = []
    
    if enable_syntax:
        verifiers.append(SyntaxVerifier())
    
    if enable_execution and driver:
        verifiers.append(ExecutionVerifier(driver))
    elif enable_execution:
        logger.warning("Execution verification requested but no driver provided")
    
    if enable_llm and llm:
        verifiers.append(LLMVerifier(llm))
    elif enable_llm:
        logger.warning("LLM verification requested but no LLM provided")
    
    return CypherVerificationPipeline(verifiers)


# =============================================================================
# CORRECTION CLASSES
# =============================================================================

class CorrectionType(Enum):
    """Types of correction methods available"""
    RULE_BASED = "rule_based"
    LLM = "llm"


@dataclass
class CorrectionResult:
    """Result of a Cypher query correction attempt"""
    corrected_query: str
    correction_type: CorrectionType
    was_modified: bool
    corrections_applied: List[str]
    error_context: Optional[str] = None
    
    def __str__(self) -> str:
        if not self.was_modified:
            return f"[{self.correction_type.value}] No corrections needed"
        corrections = ", ".join(self.corrections_applied) if self.corrections_applied else "modifications"
        return f"[{self.correction_type.value}] Applied: {corrections}"


class CypherCorrector(ABC):
    """Abstract base class for Cypher query correctors"""
    
    @property
    @abstractmethod
    def correction_type(self) -> CorrectionType:
        pass
    
    @abstractmethod
    def correct(self, cypher_query: str, error_message: str = None, schema: str = None, **kwargs) -> CorrectionResult:
        pass


class RuleBasedCorrector(CypherCorrector):
    """Rule-based correction for common issues - fast pattern-based fixes"""
    
    @property
    def correction_type(self) -> CorrectionType:
        return CorrectionType.RULE_BASED
    
    DIRECTION_FIXES = [
        (r'-\[:ACTED_IN\]->', r'<-[:ACTED_IN]-', "Actor acts IN movie (incoming)"),
        (r'-\[:DIRECTED\]->', r'<-[:DIRECTED]-', "Director directs movie (incoming)"),
        (r'-\[:WROTE\]->', r'<-[:WROTE]-', "Writer writes movie (incoming)"),
        (r'-\[:PRODUCED\]->', r'<-[:PRODUCED]-', "Producer produces movie (incoming)"),
    ]
    
    SYNTAX_FIXES = [
        (r'  +', ' ', "Remove extra spaces"),
        (r'\(:(\w)', r'(:\1', "Fix label spacing"),
        (r';\s*$', '', "Remove trailing semicolon"),
        (r'\bRETURNS\b', 'RETURN', "Fix RETURNS typo"),
        (r'\bMATCHES\b', 'MATCH', "Fix MATCHES typo"),
        (r'\bWHERES\b', 'WHERE', "Fix WHERES typo"),
    ]
    
    def correct(self, cypher_query: str, error_message: str = None, schema: str = None, **kwargs) -> CorrectionResult:
        corrected = cypher_query
        corrections_applied = []
        
        for pattern, replacement, description in self.SYNTAX_FIXES:
            if re.search(pattern, corrected):
                corrected = re.sub(pattern, replacement, corrected)
                corrections_applied.append(description)
        
        if error_message and ('direction' in error_message.lower() or 'relationship' in error_message.lower()):
            for pattern, replacement, description in self.DIRECTION_FIXES:
                if re.search(pattern, corrected, re.IGNORECASE):
                    corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                    corrections_applied.append(description)
        
        corrected, quote_fix = self._fix_unbalanced_quotes(corrected)
        if quote_fix:
            corrections_applied.append(quote_fix)
        
        if 'MATCH' in corrected.upper() and 'RETURN' not in corrected.upper():
            if not any(kw in corrected.upper() for kw in ['CREATE', 'DELETE', 'SET', 'MERGE', 'REMOVE']):
                match_vars = self._extract_variables(corrected)
                if match_vars:
                    corrected = f"{corrected.rstrip()} RETURN {match_vars[0]}"
                    corrections_applied.append(f"Added RETURN {match_vars[0]}")
        
        was_modified = corrected != cypher_query
        if was_modified:
            logger.debug(f"Rule-based corrections applied: {corrections_applied}")
        
        return CorrectionResult(
            corrected_query=corrected,
            correction_type=self.correction_type,
            was_modified=was_modified,
            corrections_applied=corrections_applied,
            error_context=error_message
        )
    
    def _fix_unbalanced_quotes(self, query: str) -> tuple:
        single_count = query.count("'") - query.count("\\'")
        double_count = query.count('"') - query.count('\\"')
        if single_count % 2 != 0 or double_count % 2 != 0:
            return query, None
        return query, None
    
    def _extract_variables(self, query: str) -> List[str]:
        pattern = r'[\(\[]\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::|[\)\]])'
        matches = re.findall(pattern, query)
        return list(dict.fromkeys(matches))


class LLMCorrector(CypherCorrector):
    """LLM-based correction - intelligent rewriting with error context"""
    
    @property
    def correction_type(self) -> CorrectionType:
        return CorrectionType.LLM
    
    CORRECTION_PROMPT = """You are a Cypher query expert. Fix the following invalid Cypher query.

## Database Schema
{schema}

## Original Query
```cypher
{query}
```

## Error Message
{error}

## Instructions
1. Analyze the error and the query
2. Fix the specific issues mentioned in the error
3. Ensure the corrected query is valid against the schema
4. Keep the query's intent the same - only fix errors

## Response Format
Respond with ONLY the corrected Cypher query. No explanation, no markdown code blocks, just the raw Cypher query."""

    def __init__(self, llm=None):
        self.llm = llm
    
    def set_llm(self, llm):
        self.llm = llm
    
    def correct(self, cypher_query: str, error_message: str = None, schema: str = None, **kwargs) -> CorrectionResult:
        llm = kwargs.get('llm', self.llm)
        
        if llm is None:
            logger.warning("No LLM available for correction")
            return CorrectionResult(
                corrected_query=cypher_query,
                correction_type=self.correction_type,
                was_modified=False,
                corrections_applied=[],
                error_context="No LLM available"
            )
        
        try:
            prompt = self.CORRECTION_PROMPT.format(
                schema=schema or "Schema not available",
                query=cypher_query,
                error=error_message or "Query validation failed"
            )
            
            response = llm.invoke(prompt)
            corrected = response.content if hasattr(response, 'content') else str(response)
            corrected = corrected.strip()
            
            if corrected.startswith('```'):
                lines = corrected.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                corrected = '\n'.join(lines).strip()
            
            was_modified = corrected != cypher_query
            if was_modified:
                logger.debug(f"LLM correction applied, query changed from {len(cypher_query)} to {len(corrected)} chars")
            
            return CorrectionResult(
                corrected_query=corrected,
                correction_type=self.correction_type,
                was_modified=was_modified,
                corrections_applied=["LLM rewrite"] if was_modified else [],
                error_context=error_message
            )
            
        except Exception as e:
            logger.error(f"LLM correction error: {e}")
            return CorrectionResult(
                corrected_query=cypher_query,
                correction_type=self.correction_type,
                was_modified=False,
                corrections_applied=[],
                error_context=f"LLM error: {str(e)}"
            )


class CypherCorrectionPipeline:
    """Pipeline for running multiple correctors (fast to thorough)"""
    
    def __init__(self, correctors: List[CypherCorrector] = None, stop_after_first_fix: bool = False):
        self.correctors = correctors or [RuleBasedCorrector()]
        self.stop_after_first_fix = stop_after_first_fix
    
    def add_corrector(self, corrector: CypherCorrector):
        self.correctors.append(corrector)
    
    def correct(self, cypher_query: str, error_message: str = None, schema: str = None, **kwargs) -> tuple:
        results = []
        current_query = cypher_query
        
        for corrector in self.correctors:
            result = corrector.correct(current_query, error_message=error_message, schema=schema, **kwargs)
            results.append(result)
            
            if result.was_modified:
                current_query = result.corrected_query
                logger.debug(f"Correction from {corrector.correction_type.value}: {result.corrections_applied}")
                if self.stop_after_first_fix:
                    break
        
        return current_query, results
    
    def get_correction_summary(self, results: List[CorrectionResult]) -> str:
        corrections = []
        for r in results:
            if r.was_modified:
                corrections.extend(r.corrections_applied)
        return ", ".join(corrections) if corrections else "No corrections applied"


def create_correction_pipeline(enable_rule_based: bool = True, enable_llm: bool = True, llm=None) -> CypherCorrectionPipeline:
    """Factory function to create a correction pipeline"""
    correctors = []
    
    if enable_rule_based:
        correctors.append(RuleBasedCorrector())
    
    if enable_llm:
        correctors.append(LLMCorrector(llm))
    
    return CypherCorrectionPipeline(correctors)


# =============================================================================
# TEXT2CYPHER RETRIEVER
# =============================================================================

# Neo4j configuration
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USER = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DB = os.environ.get('CLIENT_NEO4J_DATABASE')

# Initialize components with centralized configuration
SEED = 42
embeddings = get_neo4j_embeddings()

# Use Text2Cypher-specific LLM configuration (falls back to main LLM if not configured)
text2cypher_llm = get_text2cypher_llm()


@dataclass
class RefinementAttempt:
    """Record of a single refinement iteration"""
    iteration: int
    cypher_query: str
    verification_passed: bool
    verification_errors: List[str]
    corrections_applied: List[str]
    
    def __str__(self) -> str:
        status = "✓" if self.verification_passed else "✗"
        return f"[Iteration {self.iteration}] {status} - Errors: {len(self.verification_errors)}, Corrections: {len(self.corrections_applied)}"


class Text2CypherRAGRetriever:
    """Text2Cypher retriever with natural language to Cypher conversion and iterative refinement"""
    
    def __init__(self):
        self.embeddings = embeddings
        self.llm = text2cypher_llm
        self.neo4j_uri = NEO4J_URI
        self.neo4j_user = NEO4J_USER
        self.neo4j_password = NEO4J_PASSWORD
        self.neo4j_db = NEO4J_DB
        
        # Load refinement configuration
        self.config = get_model_config()
        self.enable_refinement = self.config.text2cypher_enable_refinement
        self.max_iterations = self.config.text2cypher_max_iterations
        
        # Log Text2Cypher configuration
        logger.info(f"Text2Cypher LLM - Provider: {self.config.effective_text2cypher_provider.value}, "
                   f"Model: {self.config.effective_text2cypher_model.value}")
        logger.info(f"Iterative refinement: {'enabled' if self.enable_refinement else 'disabled'}, "
                   f"Max iterations: {self.max_iterations}")
        
        # Initialize LLM for verification/correction (use same model as Text2Cypher)
        self.langchain_llm = get_text2cypher_langchain_llm()
        
        # Few-shot examples - ALWAYS use chunk text search for specific company/data questions
        self.examples = [
            "USER INPUT: 'What city is NovaGrid Energy Corporation headquartered in?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'NovaGrid' RETURN c.text LIMIT 10",
            "USER INPUT: 'What year is AlTahadi Aviation Group scheduled to take its inaugural flight?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'AlTahadi' RETURN c.text LIMIT 10", 
            "USER INPUT: 'Where is AtlasVentures headquartered?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'AtlasVentures' RETURN c.text LIMIT 10",
            "USER INPUT: 'What is the revenue of NovaGrid?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'NovaGrid' RETURN c.text LIMIT 10",
            "USER INPUT: 'How many Boeing aircraft does AlTahadi have?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'Boeing' OR c.text CONTAINS 'aircraft' RETURN c.text LIMIT 10",
            "USER INPUT: 'Which system must integrate with SAP Concur?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'SAP Concur' OR c.text CONTAINS 'integration' RETURN c.text LIMIT 10",
            "USER INPUT: 'What jobs will be created by 2030?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS '2030' OR c.text CONTAINS 'jobs' RETURN c.text LIMIT 10",
            "USER INPUT: 'Which RFP mentions virtual accounts?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'virtual account' OR c.text CONTAINS 'Virtual Account' RETURN c.text LIMIT 10",
            "USER INPUT: 'When are presentations scheduled?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'presentation' OR c.text CONTAINS 'August' RETURN c.text LIMIT 10",
            "USER INPUT: 'What is AtlasVentures proposal deadline?' QUERY: MATCH (c:Chunk) WHERE c.text CONTAINS 'AtlasVentures' OR c.text CONTAINS 'deadline' RETURN c.text LIMIT 10"
        ]
        
        # Extract Neo4j schema
        self.neo4j_schema = self._extract_schema()
    
    def _extract_schema(self) -> str:
        """Dynamically extract the Neo4j schema from the actual database"""
        try:
            graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                database=self.neo4j_db,
                enhanced_schema=True
            )
            graph.refresh_schema()
            schema = graph.schema
            logger.info(f"Text2Cypher - Schema extracted: {len(schema)} characters")
            return schema
        except Exception as e:
            logger.error(f"Error extracting schema: {e}")
            return "Schema extraction failed"
    
    def _generate_cypher(self, query: str, driver) -> Tuple[str, Any]:
        """Generate Cypher query from natural language using Text2CypherRetriever"""
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=self.llm,
            neo4j_schema=self.neo4j_schema,
            examples=self.examples,
            neo4j_database=self.neo4j_db
        )
        
        search_results = retriever.search(query_text=query)
        
        cypher_query = None
        if hasattr(search_results, 'metadata') and search_results.metadata:
            cypher_query = search_results.metadata.get('cypher_query') or search_results.metadata.get('cypher')
        
        if not cypher_query and hasattr(retriever, '_last_cypher_query'):
            cypher_query = retriever._last_cypher_query
            
        return cypher_query, search_results
    
    def _iterative_refinement(
        self,
        initial_cypher: str,
        driver,
        query: str
    ) -> Tuple[str, List[RefinementAttempt], bool]:
        """Apply iterative refinement to a Cypher query"""
        if not self.enable_refinement:
            logger.debug("Iterative refinement disabled")
            return initial_cypher, [], True
        
        if not initial_cypher:
            logger.warning("No Cypher query to refine")
            return initial_cypher, [], False
        
        attempts = []
        current_cypher = initial_cypher
        
        # Create verification pipeline from config
        verifiers = self.config.text2cypher_verifiers or ['syntax', 'execution']
        verification_pipeline = create_verification_pipeline(
            enable_syntax='syntax' in verifiers,
            enable_execution='execution' in verifiers,
            enable_llm='llm' in verifiers,
            driver=driver,
            llm=self.langchain_llm if 'llm' in verifiers else None
        )
        logger.debug(f"Verification pipeline configured with: {verifiers}")
        
        # Create correction pipeline from config
        correctors = self.config.text2cypher_correctors or ['rule_based', 'llm']
        correction_pipeline = create_correction_pipeline(
            enable_rule_based='rule_based' in correctors,
            enable_llm='llm' in correctors,
            llm=self.langchain_llm if 'llm' in correctors else None
        )
        logger.debug(f"Correction pipeline configured with: {correctors}")
        
        for iteration in range(1, self.max_iterations + 1):
            logger.debug(f"Refinement iteration {iteration}/{self.max_iterations}")
            
            is_valid, verification_results = verification_pipeline.verify(
                current_cypher,
                schema=self.neo4j_schema,
                driver=driver,
                database=self.neo4j_db
            )
            
            error_messages = [
                r.error_message for r in verification_results 
                if not r.is_valid and r.error_message
            ]
            
            attempt = RefinementAttempt(
                iteration=iteration,
                cypher_query=current_cypher,
                verification_passed=is_valid,
                verification_errors=error_messages,
                corrections_applied=[]
            )
            
            if is_valid:
                logger.info(f"Cypher query passed verification at iteration {iteration}")
                attempts.append(attempt)
                return current_cypher, attempts, True
            
            logger.debug(f"Verification failed: {error_messages}")
            
            error_context = verification_pipeline.get_error_summary(verification_results)
            
            corrected_cypher, correction_results = correction_pipeline.correct(
                current_cypher,
                error_message=error_context,
                schema=self.neo4j_schema,
                llm=self.langchain_llm
            )
            
            corrections = []
            for cr in correction_results:
                if cr.was_modified:
                    corrections.extend(cr.corrections_applied)
            
            attempt.corrections_applied = corrections
            attempts.append(attempt)
            
            if corrected_cypher == current_cypher:
                logger.warning("No corrections could be applied, stopping refinement")
                break
            
            current_cypher = corrected_cypher
            logger.debug(f"Applied corrections: {corrections}")
        
        logger.warning(f"Refinement completed after {len(attempts)} iterations without full validation")
        return current_cypher, attempts, False
    
    def search(self, query: str) -> Dict[str, Any]:
        """Neo4j Text2Cypher query with natural language to Cypher conversion and iterative refinement"""
        
        with neo4j.GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password), database=self.neo4j_db) as driver:
            logger.info(f"Executing Text2Cypher query for: {query}")
            
            try:
                retriever = Text2CypherRetriever(
                    driver=driver,
                    llm=self.llm,
                    neo4j_schema=self.neo4j_schema,
                    examples=self.examples,
                    neo4j_database=self.neo4j_db
                )
                
                logger.debug("Text2Cypher retriever initialized successfully")
                
                search_results = retriever.search(query_text=query)
                
                generated_cypher = None
                if hasattr(search_results, 'metadata') and search_results.metadata:
                    generated_cypher = search_results.metadata.get('cypher_query') or search_results.metadata.get('cypher')
                    if generated_cypher:
                        logger.debug(f"Generated Cypher: {generated_cypher}")
                    else:
                        logger.debug(f"Metadata keys: {list(search_results.metadata.keys())}")
                
                if not generated_cypher and hasattr(retriever, '_last_cypher_query'):
                    generated_cypher = retriever._last_cypher_query
                    logger.debug(f"Last Cypher query: {generated_cypher}")
                
                refinement_attempts = []
                refinement_success = True
                
                if self.enable_refinement and generated_cypher:
                    refined_cypher, refinement_attempts, refinement_success = self._iterative_refinement(
                        generated_cypher,
                        driver,
                        query
                    )
                    
                    if refinement_attempts:
                        logger.info(f"Refinement: {len(refinement_attempts)} iterations, success={refinement_success}")
                        for attempt in refinement_attempts:
                            logger.debug(str(attempt))
                    
                    if refined_cypher != generated_cypher and refinement_success:
                        logger.info("Re-executing refined Cypher query")
                        try:
                            with driver.session(database=self.neo4j_db) as session:
                                result = session.run(refined_cypher)
                                records = list(result)
                                search_results = records
                        except Exception as e:
                            logger.warning(f"Failed to execute refined query: {e}, using original results")
                
                logger.debug(f"Text2Cypher search completed, type: {type(search_results)}")
                
                if hasattr(search_results, 'items'):
                    items = search_results.items
                    logger.debug(f"Found {len(items)} items in search_results.items")
                else:
                    items = search_results if isinstance(search_results, list) else [search_results]
                    logger.debug(f"Using direct results, type: {type(items)}, length: {len(items) if hasattr(items, '__len__') else 'unknown'}")
                
                logger.info(f"Text2Cypher retrieved {len(items) if hasattr(items, '__len__') else 'unknown'} results")
                
                retrieval_details = []
                context_parts = []
                
                for i, item in enumerate(items, 1):
                    logger.debug(f"Processing item {i}: {type(item)}")
                    
                    if hasattr(item, 'content'):
                        content = item.content
                    elif isinstance(item, dict):
                        content = str(item)
                    elif hasattr(item, 'data'):
                        content = str(dict(item))
                    else:
                        content = str(item)
                    
                    context_parts.append(f"Result {i}:\n{content}")
                    retrieval_details.append({
                        'content': content,
                        'source': 'Text2Cypher Query',
                        'type': 'cypher_result'
                    })
                
                if not context_parts:
                    logger.warning("No context parts found!")
                    return {
                        'method': 'Text2Cypher + LLM',
                        'query': query,
                        'final_answer': 'No results found using Text2Cypher approach.',
                        'retrieved_chunks': 0,
                        'retrieval_details': [],
                        'refinement_iterations': len(refinement_attempts),
                        'refinement_success': False
                    }
                
                context = "\n\n".join(context_parts)
                logger.debug(f"Combined context length: {len(context)} characters")
                
                prompt = f"""Based on the following query results from a knowledge graph database, provide a comprehensive answer to the question.

Question: {query}

Query Results:
{context}

Instructions:
1. Use the information from the query results to answer the question directly
2. If the results contain the exact answer, state it clearly
3. If the results are partial or need interpretation, explain what you found
4. If the results don't contain enough information to answer the question, state this clearly
5. Be factual and only use information present in the results
6. Format your response clearly and concisely

Please provide a factual, well-structured response."""

                try:
                    llm_response = self.llm.invoke(prompt)
                    final_answer = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                    logger.debug(f"LLM response generated: {len(final_answer)} characters")
                except Exception as e:
                    final_answer = f"Error generating LLM response: {e}"
                    logger.error(f"LLM error: {e}")
                
                return {
                    'method': 'Text2Cypher + LLM',
                    'query': query,
                    'final_answer': final_answer,
                    'retrieved_chunks': len(items) if hasattr(items, '__len__') else 0,
                    'retrieval_details': retrieval_details,
                    'generated_cypher': generated_cypher,
                    'refinement_iterations': len(refinement_attempts),
                    'refinement_success': refinement_success
                }
                
            except Exception as e:
                logger.error(f"Text2Cypher error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'method': 'Text2Cypher + LLM',
                    'query': query,
                    'final_answer': f"Error with Text2Cypher processing: {e}",
                    'retrieved_chunks': 0,
                    'retrieval_details': [],
                    'refinement_iterations': 0,
                    'refinement_success': False
                }


# Factory function for easy instantiation
def create_text2cypher_retriever() -> Text2CypherRAGRetriever:
    """Create a Text2Cypher retriever instance"""
    return Text2CypherRAGRetriever()


# Main interface function for integration with benchmark system
def query_text2cypher_rag(query: str, **kwargs) -> Dict[str, Any]:
    """Text2Cypher RAG retrieval with natural language to Cypher conversion"""
    try:
        retriever = create_text2cypher_retriever()
        result = retriever.search(query)
        
        return {
            'final_answer': result['final_answer'],
            'retrieval_details': [
                {
                    'content': detail['content'],
                    'metadata': {'source': detail['source'], 'type': detail['type']}
                } for detail in result['retrieval_details']
            ],
            'method': 'text2cypher_rag',
            'performance_metrics': {
                'retrieved_chunks': result['retrieved_chunks'],
                'completion_time': 0,
                'llm_calls': 1 + result.get('refinement_iterations', 0),
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'refinement_iterations': result.get('refinement_iterations', 0),
                'refinement_success': result.get('refinement_success', True)
            }
        }
        
    except Exception as e:
        logger.error(f"Error in Text2Cypher retrieval: {e}")
        import traceback
        traceback.print_exc()
        return {
            'final_answer': f"Error during Text2Cypher retrieval: {str(e)}",
            'retrieval_details': [],
            'method': 'text2cypher_rag_error',
            'performance_metrics': {
                'retrieved_chunks': 0,
                'completion_time': 0,
                'llm_calls': 0,
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'refinement_iterations': 0,
                'refinement_success': False
            }
        }
