"""
Utilities module for Ax0n - Helper functions and utilities
"""

import json
import re
import structlog
from typing import Any, Dict, List, Optional

logger = structlog.get_logger(__name__)

# Default max JSON response size: 10 MB
_DEFAULT_MAX_JSON_SIZE = 10_000_000


def parse_json_object_from_response(
    response: str,
    max_size: int = _DEFAULT_MAX_JSON_SIZE,
) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the outermost JSON object from a messy LLM response.

    Finds the first '{' and last '}' in the response, extracts that substring,
    and attempts to parse it as JSON.

    Args:
        response: Raw LLM response text that may contain extra text around JSON
        max_size: Maximum allowed size in bytes for the JSON substring

    Returns:
        Parsed dictionary, or None if parsing fails
    """
    json_start = response.find('{')
    json_end = response.rfind('}')
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return None

    json_str = response[json_start:json_end + 1]

    if len(json_str) > max_size:
        logger.warning(
            "JSON response exceeds size limit",
            size=len(json_str),
            max_size=max_size,
        )
        return None

    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse JSON object from response", length=len(response))
        return None


def parse_json_array_from_response(
    response: str,
    max_size: int = _DEFAULT_MAX_JSON_SIZE,
) -> Optional[List[Any]]:
    """
    Extract and parse the outermost JSON array from a messy LLM response.

    Finds the first '[' and last ']' in the response, extracts that substring,
    and attempts to parse it as JSON.

    Args:
        response: Raw LLM response text that may contain extra text around JSON
        max_size: Maximum allowed size in bytes for the JSON substring

    Returns:
        Parsed list, or None if parsing fails
    """
    json_start = response.find('[')
    json_end = response.rfind(']')
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        return None

    json_str = response[json_start:json_end + 1]

    if len(json_str) > max_size:
        logger.warning(
            "JSON array response exceeds size limit",
            size=len(json_str),
            max_size=max_size,
        )
        return None

    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
        return None
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse JSON array from response", length=len(response))
        return None


def extract_claims(text: str, min_words: int = 4, min_chars: int = 15, max_claims: int = 5) -> List[str]:
    """
    Extract declarative factual claims from text using sentence boundary
    detection and information density filtering.

    Shared utility used by both Grounding Validator and Memory Extractor
    to avoid duplicating claim extraction logic.

    Args:
        text: Input text to extract claims from
        min_words: Minimum word count per claim
        min_chars: Minimum character length per claim
        max_claims: Maximum claims to return

    Returns:
        List of extracted claim strings
    """
    content = text.strip()
    if not content:
        return []

    # Regex-based sentence splitting on ., !, ? followed by whitespace + uppercase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', content)

    # Fallback: try simple period splitting
    if len(sentences) == 1 and sentences[0] == content:
        sentences = [s.strip() for s in content.split('.') if s.strip()]

    # Non-declarative indicators
    non_declarative = {'should', 'could', 'might', 'would', 'perhaps', 'maybe'}

    claims = []
    for s in sentences:
        s = s.strip().rstrip('.')
        if not s:
            continue
        if s.endswith('?'):
            continue
        words = s.split()
        if len(words) < min_words:
            continue
        if len(s) < min_chars:
            continue
        first_word = words[0].lower()
        if first_word in non_declarative:
            continue
        prefix = ' '.join(words[:2]).lower()
        if prefix in ("let us", "let's"):
            continue
        claims.append(s)

    return claims[:max_claims]


from .embeddings import EmbeddingProvider

__all__ = [
    'parse_json_object_from_response',
    'parse_json_array_from_response',
    'extract_claims',
    'EmbeddingProvider',
]
