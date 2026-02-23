import re
from typing import List
from ..interfaces import NerEntity

class DeterministicEnricher:
    """
    Implements rule-based and regex-based entity extraction as a backstop or boost to the model.
    Supports Hebrew, Arabic, Persian, and English.
    """
    
    PATTERNS = {
        "money": {
            "he": r'(?:₪|\$|€|£)\s?\d+(?:[.,]\d+)?(?:\s*(?:מיליון|אלף|מיליארד|M|K|B|NIS))?|\b\d+(?:[.,]\d+)?\s*(?:₪|אירו|דולר|ש"ח|מיליון|אלף)\b',
            "ar": r'(?:\$|€|£)\s?\d+(?:[.,]\d+)?(?:\s*[KMB])?|\b\d+(?:[.,]\d+)?\s*(?:دولار|ريال|درهم|يورو|جنيه|مليون|مليار|ألف|JOD)\b',
            "fa": r'(?:\$|€|£)\s?\d+(?:[.,]\d+)?(?:\s*[KMB])?|\b\d+(?:[.,]\d+)?\s*(?:ریال|تومان|دلار|یورو|میلیون|میلیارد|هزار)\b',
            "en": r'(?:\$|€|£|¥)\s?\d+(?:[.,]\d+)?(?:\s*[KMB])?|\b\d+(?:[.,]\d+)?\s*(?:USD|EUR|GBP|million|billion|trillion)\b'
        },
        "percent": {
            "en": r'\b\d+(?:[.,]\d+)?%\b',
            "ar": r'\b[0-9٠-٩]+(?:[.,][0-9٠-٩]+)?%\b',
            "fa": r'\b[۰-۹0-9]+(?:[\.,٫][۰-۹0-9]+)?٪\b',
            "he": r'\b\d+(?:[.,]\d+)?%\b'
        },
        "date_time": r'\b(19|20)\d{2}\b|\b(۱۹|۲۰)[۰-۹]{2}\b|\b(١٩|٢٠)[٠-٩]{2}\b',
        "transport_id": r'\b[A-Z]{1,3}\d{2,4}\b',
        "project": r'\b(?:Project|Proj|פרויקט|משروع|پروژه)\s*[-:]?\s*[A-Z0-9][A-Z0-9\-_/\.]{4,}(?:\s*v\d+(?:\.\\d+)*)?\b',
        "airport_facility": {
            "he": r'(?:נמל\s+התעופה\s+[^,\n]+|שדה\s+התעופה\s+[^,\n]+|מתחם\s+[^,\n]+|קניון\s+[^,\n]+)',
            "ar": r'(?:مطار\s+[^,\n]+(?:الدولي)?|محطة\s+[^,\n]+|مجمع\s+[^,\n]+)',
            "fa": r'(?:فرودگاه\s+[^,\n]+|ایستگاه\s+[^,\n]+|مجموعه\s+[^,\n]+|ایران\s*مال|ایرانمال)',
            "en": r'(?:[A-Za-z\- ]+\s+International\s+Airport|Airport\s+of\s+[A-Za-z\- ]+|Terminal\s+\d+)'
        },
        "digital_indicator": r'\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-fA-F0-9]{32,64}\b',
        "vulnerability": r'\bCVE-\d{4}-\d{4,}\b'
    }

    @staticmethod
    async def enrich(entities: List[NerEntity], text: str, language: str) -> List[NerEntity]:
        """
        Runs deterministic extractors and adds entities if not already covered by overlaps.
        """
        augmented = list(entities)
        
        # 1. Money Enrichment
        money_pattern = DeterministicEnricher.PATTERNS["money"].get(language, DeterministicEnricher.PATTERNS["money"]["en"])
        for m in re.finditer(money_pattern, text):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="money", start=m.start(), end=m.end(), score=0.95, source="regex"
                ))
        
        # 2. Percent Enrichment
        percent_pattern = DeterministicEnricher.PATTERNS["percent"].get(language, DeterministicEnricher.PATTERNS["percent"]["en"])
        for m in re.finditer(percent_pattern, text):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="percent", start=m.start(), end=m.end(), score=0.95, source="regex"
                ))

        # 3. Project Enrichment
        for m in re.finditer(DeterministicEnricher.PATTERNS["project"], text, re.IGNORECASE):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="project", start=m.start(), end=m.end(), score=0.95, source="regex"
                ))

        # 4. Year/Date Enrichment (V4: date_time)
        for m in re.finditer(DeterministicEnricher.PATTERNS["date_time"], text):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="date_time", start=m.start(), end=m.end(), score=0.90, source="regex"
                ))
                
        # 5. Transport ID Enrichment (V4: transport_id)
        for m in re.finditer(DeterministicEnricher.PATTERNS["transport_id"], text):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="transport_id", start=m.start(), end=m.end(), score=0.90, source="regex"
                ))
                
        # 6. Airport/Mall Facility Enrichment
        airport_pattern = DeterministicEnricher.PATTERNS["airport_facility"].get(language)
        if airport_pattern:
             for m in re.finditer(airport_pattern, text):
                if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                    augmented.append(NerEntity(
                        text=m.group(), label="facility", start=m.start(), end=m.end(), score=0.90, source="regex"
                    ))
                    
        # 7. Cyber-Intel indicators
        for m in re.finditer(DeterministicEnricher.PATTERNS["digital_indicator"], text):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="digital_indicator", start=m.start(), end=m.end(), score=0.95, source="regex"
                ))
                
        for m in re.finditer(DeterministicEnricher.PATTERNS["vulnerability"], text):
            if not DeterministicEnricher._is_covered(m.start(), m.end(), augmented):
                augmented.append(NerEntity(
                    text=m.group(), label="vulnerability", start=m.start(), end=m.end(), score=0.98, source="regex"
                ))

        return augmented

    @staticmethod
    def _is_covered(start: int, end: int, entities: List[NerEntity]) -> bool:
        """Checks if a span overlaps significantly with existing entities."""
        for ent in entities:
            # Check for partial or full overlap
            if max(start, ent.start) < min(end, ent.end):
                return True
        return False
