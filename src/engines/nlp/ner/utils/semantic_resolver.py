import logging
import re
from typing import Dict, List, Any, Optional, Set

class SemanticResolver:
    """
    Expert-grade semantic resolver for domains and labels.
    Uses precise alias matching and fuzzy similarity (Levenshtein) to map 
    arbitrary user inputs to standardized configuration keys.
    """
    
    def __init__(self, registry: Dict[str, List[str]], threshold: float = 0.8):
        """
        registry: A mapping of {canonical_key: [list_of_aliases]}
        """
        self.registry = registry
        self.threshold = threshold
        self._logger = logging.getLogger(__name__)

    def resolve_multiple(self, queries: List[str]) -> List[str]:
        """Resolves a list of queries to their canonical forms, skipping unknowns."""
        if not queries:
            return []
        
        resolved = set()
        for q in queries:
            res = self.resolve(q)
            if res:
                resolved.add(res)
        return list(resolved)

    def resolve(self, query: str) -> Optional[str]:
        """Resolves a single query string to a canonical key."""
        if not query:
            return None
            
        q_norm = query.lower().strip()
        
        # 1. Direct match (Canonical Key)
        if q_norm in self.registry:
            return q_norm
            
        # 2. Alias match (Exact)
        for canonical, aliases in self.registry.items():
            if q_norm in [a.lower() for a in aliases]:
                return canonical
                
        # 3. Fuzzy match (Levenshtein)
        best_match = None
        highest_score = 0.0
        
        for canonical, aliases in self.registry.items():
            # Check canonical itself
            score = self._similarity(q_norm, canonical.lower())
            if score > highest_score:
                highest_score = score
                best_match = canonical
            
            # Check aliases
            for alias in aliases:
                score = self._similarity(q_norm, alias.lower())
                if score > highest_score:
                    highest_score = score
                    best_match = canonical
                    
        if highest_score >= self.threshold:
            self._logger.info(f"Semantically resolved '{query}' -> '{best_match}' (score: {highest_score:.2f})")
            return best_match
            
        self._logger.debug(f"Failed to resolve '{query}' semantically. Best score: {highest_score:.2f}")
        return None

    def _similarity(self, s1: str, s2: str) -> float:
        """Standard Levenshtein similarity algorithm."""
        if s1 == s2: return 1.0
        n, m = len(s1), len(s2)
        if n == 0 or m == 0: return 0.0
        
        if n < m:
            s1, s2 = s2, s1
            n, m = m, n

        current_row = range(m + 1)
        for i in range(1, n + 1):
            previous_row, current_row = current_row, [i] + [0] * m
            for j in range(1, m + 1):
                add, delete, change = previous_row[j] + 1, current_row[j-1] + 1, previous_row[j-1]
                if s1[i-1] != s2[j-1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        lev_dist = current_row[m]
        return 1.0 - (lev_dist / n)
