import json
import os
from typing import Optional, Tuple
from datetime import datetime

CACHE_FILE = "qa_cache.json"
MAX_CACHE_SIZE = 1000  # Maximum number of cached questions
MIN_SIMILARITY_THRESHOLD = 0.85  # Increased threshold for more strict matching

def load_cache() -> dict:
    """Load the cache from file if it exists, otherwise return empty dict."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_cache(cache: dict):
    """Save the cache to file."""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_cached_response(question: str) -> Optional[str]:
    """Get cached response for a question if it exists."""
    cache = load_cache()
    return cache.get(question)

def cache_response(question: str, answer: str):
    """Cache a question-answer pair with additional metadata."""
    cache = load_cache()
    
    # Add metadata to the answer
    cache_entry = {
        "answer": answer,
        "timestamp": datetime.now().isoformat(),
        "question_length": len(question),
        "answer_length": len(answer)
    }
    
    # If cache is full, remove oldest entry
    if len(cache) >= MAX_CACHE_SIZE:
        oldest_key = min(cache.keys(), key=lambda k: cache[k].get("timestamp", ""))
        del cache[oldest_key]
    
    cache[question] = cache_entry
    save_cache(cache)

def get_similar_question(question: str, threshold: float = MIN_SIMILARITY_THRESHOLD) -> Optional[Tuple[str, str]]:
    """
    Check if there's a similar question in cache with additional validation.
    Returns tuple of (similar_question, answer) if found, None otherwise.
    """
    from difflib import SequenceMatcher
    
    cache = load_cache()
    best_match = None
    best_score = 0
    
    for cached_q, cache_data in cache.items():
        # Calculate similarity
        similarity = SequenceMatcher(None, question.lower(), cached_q.lower()).ratio()
        
        # Additional validation checks
        if similarity >= threshold:
            # Check if the cached answer is still valid
            answer = cache_data.get("answer", "")
            if not answer or len(answer) < 10:  # Skip very short answers
                continue
                
            # Update best match if this is more similar
            if similarity > best_score:
                best_score = similarity
                best_match = (cached_q, answer)
    
    return best_match

def get_cache_stats() -> dict:
    """Get statistics about the cache."""
    cache = load_cache()
    return {
        "total_questions": len(cache),
        "average_answer_length": sum(len(data.get("answer", "")) for data in cache.values()) / len(cache) if cache else 0,
        "oldest_entry": min((data.get("timestamp", "") for data in cache.values()), default=""),
        "newest_entry": max((data.get("timestamp", "") for data in cache.values()), default="")
    } 