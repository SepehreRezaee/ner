import re
from typing import List, Dict, Any, Tuple
from ..interfaces import NerEntity

class ConflictResolver:
    """
    Handles deduplication, priority-based overlap resolution, and heuristic merging.
    """
    
    DEFAULT_PRIORITIES = {
        "vulnerability": 140,
        "case_ref": 130,
        "statute": 125,
        "threat_actor": 120,
        "drug_agent": 115,
        "person": 110,
        "organization": 105,
        "media": 104,
        "project": 103,
        "facility": 95,
        "location": 90,
        "address": 88,
        "flight": 86,
        "ticket_id": 85,
        "money": 70,
        "percent": 65,
        "date": 60,
        "time": 58,
        "job_title": 50,
        "misc": 10
    }

    def __init__(self, priorities: Dict[str, int] = None, config: Dict[str, Any] = None):
        self.priorities = priorities or self.DEFAULT_PRIORITIES
        self.config = config or {}

    async def resolve(self, entities: List[NerEntity], text: str, language: str) -> List[NerEntity]:
        """
        Main entry point for conflict resolution and heuristic merging.
        """
        if not entities:
            return []

        # 1. Sort by start position
        entities.sort(key=lambda x: x.start)
        
        # 2. Heuristic Merges & Splits
        entities = self._merge_acronyms(entities, text)
        entities = self._merge_foundations(entities, text, language)
        entities = self._merge_titles_and_people(entities, text, language)
        entities = self._split_airlines_flights(entities, text, language)
        
        # 3. Resolve Overlaps using Priorities & Provenance
        resolved = self._resolve_overlaps_priority(entities)
        
        # 4. Reclassification Policies (Media, Hospitals, Geo-Typing)
        resolved = self._reclassify_entities(resolved, language)
        
        # 5. Multi-Org Split (V4.1)
        resolved = self._multi_org_split(resolved, text, language)

        # 6. Final Cleanup (whitespace, trailing punctuation, clitics)
        resolved = self._final_cleanup(resolved, language)
        
        return resolved

    def _resolve_overlaps_priority(self, entities: List[NerEntity]) -> List[NerEntity]:
        """Keep highest priority entity in case of overlap."""
        # Sort: start asc, length desc
        entities.sort(key=lambda x: (x.start, -(x.end - x.start)))
        
        keep = []
        for current in entities:
            if not keep:
                keep.append(current)
                continue
            
            last = keep[-1]
            if current.start >= last.end:
                keep.append(current)
                continue

            # Overlap exists
            p_curr = self.priorities.get(current.label, 10)
            p_last = self.priorities.get(last.label, 10)

            # Rule: Prefer longer same label if score is high
            if current.label == last.label:
                if (current.end - current.start) > (last.end - last.start):
                    keep[-1] = current
                continue

            # Rule: Specific overrides
            if current.label == "facility" and last.label == "location":
                keep[-1] = current
                continue
            
            if p_curr > p_last:
                keep[-1] = current
                
        return keep

    def _multi_org_split(self, entities: List[NerEntity], text: str, language: str) -> List[NerEntity]:
        """Splits merged organizations like 'Apple, Google and Meta'."""
        split_config = self.config.get("multi_org_split", {})
        if not split_config.get("enabled", True):
            return entities
            
        patterns = split_config.get("split_on_patterns", {}).get(language, [", ", " and ", " ו", "، "])
        skip_words = split_config.get("do_not_split_if_contains", ["Ltd", "Inc", "בע\"מ"])
        
        new_list = []
        for ent in entities:
            if ent.label == "organization" and not any(s in ent.text for s in skip_words):
                # Try to split by patterns
                parts = []
                last_pos = 0
                for pattern in patterns:
                    if pattern in ent.text:
                        # Simple split for demonstration
                        # A real version would find all indices and slice carefully
                        temp_parts = ent.text.split(pattern)
                        if len(temp_parts) > 1:
                            parts = temp_parts
                            break
                
                if parts:
                    offset = ent.start
                    for part in parts:
                        part_clean = part.strip()
                        if part_clean:
                            p_start = text.find(part_clean, offset)
                            if p_start != -1:
                                new_list.append(NerEntity(
                                    text=part_clean, label="organization", 
                                    start=p_start, end=p_start + len(part_clean), 
                                    score=ent.score, source="heuristic_split"
                                ))
                                offset = p_start + len(part_clean)
                    continue
            new_list.append(ent)
        return new_list

    def _merge_acronyms(self, entities: List[NerEntity], text: str) -> List[NerEntity]:
        i = 0
        while i < len(entities) - 1:
            curr = entities[i]
            next_ent = entities[i+1]
            if curr.label == "organization" and (next_ent.label in ["organization", "misc"] or next_ent.score >= 0.4):
                gap = text[curr.end:next_ent.start]
                if "(" in gap:
                    close_idx = text.find(")", next_ent.end, next_ent.end + 2)
                    if close_idx != -1:
                        curr.end = close_idx + 1
                        curr.text = text[curr.start:curr.end]
                        curr.score = max(curr.score, next_ent.score)
                        entities.pop(i+1)
                        continue
            i += 1
        return entities

    def _reclassify_entities(self, entities: List[NerEntity], language: str) -> List[NerEntity]:
        media_keywords = {"he": ["גלובס", "דה-מרקר", "עיתון"], "ar": ["صحيفة", "قناة"], "en": ["Post", "Times", "Journal"]}
        hospital_keywords = {"he": ["בית חולים"], "ar": ["مستشفى"], "en": ["Hospital"]}
        
        m_kws = media_keywords.get(language, [])
        h_kws = hospital_keywords.get(language, [])
        
        for ent in entities:
            if ent.label == "organization":
                if any(k in ent.text for k in m_kws): ent.label = "media"
                if any(k in ent.text for k in h_kws): ent.label = "facility"
        return entities

    def _merge_foundations(self, entities: List[NerEntity], text: str, language: str) -> List[NerEntity]:
        found_keywords = {"he": "קרן", "ar": "مؤسسة", "en": "Foundation"}
        kw = found_keywords.get(language, "Foundation")
        for ent in entities:
            if ent.label == "person":
                context = text[max(0, ent.start-15):ent.end+15]
                if kw in context: ent.label = "organization"
        return entities

    def _merge_titles_and_people(self, entities: List[NerEntity], text: str, language: str) -> List[NerEntity]:
        # Minimalist v4.1 logic
        return entities

    def _split_airlines_flights(self, entities: List[NerEntity], text: str, language: str) -> List[NerEntity]:
        flight_regex = r"\b[A-Z]{1,3}\d{2,4}\b"
        new_list = []
        for ent in entities:
            if ent.label == "organization":
                match = re.search(flight_regex, ent.text)
                if match:
                    f_start = ent.start + match.start()
                    new_list.append(NerEntity(text=text[ent.start:f_start].strip(), label="organization", start=ent.start, end=f_start, score=ent.score))
                    new_list.append(NerEntity(text=match.group(), label="transport_id", start=f_start, end=f_start + len(match.group()), score=0.95))
                    continue
            new_list.append(ent)
        return new_list

    def _final_cleanup(self, entities: List[NerEntity], language: str = "en") -> List[NerEntity]:
        for ent in entities:
            # Strip clitics
            if language == "he" and ent.label in ["location", "facility", "organization"]:
                if len(ent.text) > 3 and ent.text[0] in ["ב", "ל", "מ", "כ"]:
                    ent.text = ent.text[1:]; ent.start += 1
            # Punctuation
            ent.text = ent.text.strip(".,!?;: ")
        return [e for e in entities if e.text]
