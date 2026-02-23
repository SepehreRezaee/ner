import re
import logging
from typing import List, Dict, Any
from ..interfaces import NerEntity

class QASafetySuite:
    """
    QA Layer to ensure output integrity and label sanity.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = logging.getLogger(__name__)

    async def run_checks(self, entities: List[NerEntity], original_text: str) -> List[NerEntity]:
        """Runs all enabled QA checks."""
        if not self.config or not self.config.get("enabled", True):
            return entities
            
        checked_entities = entities
        
        # 1. Offset Integrity
        if self.config.get("offset_integrity", {}).get("enabled", True):
            checked_entities = self._check_offset_integrity(checked_entities, original_text)
            
        # 2. Label Sanity
        if self.config.get("label_sanity", {}).get("enabled", True):
            checked_entities = self._check_label_sanity(checked_entities)
            
        return checked_entities

    def _check_offset_integrity(self, entities: List[NerEntity], original_text: str) -> List[NerEntity]:
        """Ensures ent.text matches original_text[ent.start:ent.end]."""
        policy = self.config.get("offset_integrity", {})
        qualified = []
        
        for ent in entities:
            actual_span = original_text[ent.start:ent.end]
            if actual_span != ent.text:
                self._logger.warning(f"QA Offset mismatch: '{ent.text}' vs '{actual_span}' at [{ent.start}:{ent.end}]")
                if policy.get("fail_if_span_text_mismatch", True):
                    # Remediation: try to re-slice or drop
                    # For V4.1 we drop to be safe, or use the orig span if it looks okay
                    if len(actual_span.strip()) > 0:
                        ent.text = actual_span
                        qualified.append(ent)
                    continue
            qualified.append(ent)
        return qualified

    def _check_label_sanity(self, entities: List[NerEntity]) -> List[NerEntity]:
        """Label-specific validation rules."""
        rules = self.config.get("label_sanity", {}).get("rules", [])
        if not rules:
            return entities
            
        qualified = []
        for ent in entities:
            is_valid = True
            for rule in rules:
                if ent.label == rule.get("label") or ent.label in rule.get("labels", []):
                    # Regex match
                    patterns = rule.get("reject_if_not_matches_any_regex", [])
                    if patterns:
                        # This would need access to enrichment regexes or standalone ones
                        # For now, minimal check
                        pass
                    
                    # Contains any
                    required_parts = rule.get("reject_if_not_contains_any", [])
                    if required_parts:
                        if not any(p in ent.text for p in required_parts):
                            is_valid = False
                            self._logger.info(f"QA Label Sanity Fail: [{ent.label}] '{ent.text}' dropped.")
                            break
            if is_valid:
                qualified.append(ent)
        return qualified
