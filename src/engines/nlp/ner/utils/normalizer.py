import re
import logging
from typing import Tuple, Dict, Any, List
from .offsets import OffsetMapper

class Normalizer:
    """
    Language-aware normalizer that uses externalized rules from NerConfig.
    Maintains offset integrity via OffsetMapper.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = logging.getLogger(__name__)

    async def normalize(self, text: str, language: str) -> Tuple[str, OffsetMapper]:
        """
        Runs normalization rules for the given language.
        Returns (normalized_text, offset_mapper).
        """
        mapper = OffsetMapper(text)
        lang_config = self.config.get(language, {})
        rules = lang_config.get("rules", [])
        
        for rule in rules:
            name = rule.get("name", "unnamed_rule")
            if "map" in rule:
                for old, new in rule["map"].items():
                    text, mapper = self._replace_all(text, mapper, old, new)
            elif "pattern" in rule:
                pattern = rule["pattern"]
                replace = rule.get("replace", "")
                text, mapper = self._replace_by_pattern(text, mapper, pattern, replace)
        
        # Post-processing: Collapse extra spaces if requested
        # (Usually handled by rules, but we can ensure here if needed)
        
        mapper.set_normalized(text)
        return text, mapper

    def _replace_all(self, text: str, mapper: OffsetMapper, old: str, new: str) -> Tuple[str, OffsetMapper]:
        """Replaced all occurrences of fixed string."""
        start_search = 0
        while True:
            idx = text.find(old, start_search)
            if idx == -1:
                break
            
            end = idx + len(old)
            mapper.replace(idx, end, new)
            text = text[:idx] + new + text[end:]
            start_search = idx + len(new)
        return text, mapper

    def _replace_by_pattern(self, text: str, mapper: OffsetMapper, pattern: str, replace: str) -> Tuple[str, OffsetMapper]:
        """Replaces all matches of a regex pattern."""
        # We must process matches carefully to keep mapper in sync
        # Best way is to find all matches, then replace from end to beginning
        # so that earlier offsets remain valid during the loop.
        
        matches = list(re.finditer(pattern, text))
        for m in reversed(matches):
            start, end = m.start(), m.end()
            # If replacement is a simple string, we use it.
            # If it were a function or had backrefs, it would be more complex.
            # For current YAML spec, it's a simple string.
            mapper.replace(start, end, replace)
            text = text[:start] + replace + text[end:]
            
        return text, mapper
