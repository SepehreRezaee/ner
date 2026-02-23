import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.engines.nlp.ner.model import build_service

async def test_production_pipeline():
    print("🚀 Initializing Production NER Pipeline...")
    config_path = Path("configs/ner.yaml")
    service = build_service(config_path)
    
    # V4.0 Omni-Domain Stress Test Cases
    test_cases = [
        {
            "name": "Legal & Hebrew Clitic",
            "text": "לפי חוק העונשין (Statute), העונש כבד. בירושלים דנים בכך.",
            "lang": "he",
            "expected": [("statute", "חוק העונשין"), ("location", "ירושלים")]
        },
        {
            "name": "Clinical & Arabic Clitic",
            "text": "تم تشخيص مرض السكري في مستشفى الملك خالد.",
            "lang": "ar",
            "expected": [("condition", "السكري"), ("facility", "مستشفى الملك خالد")]
        },
        {
            "name": "Cyber-Intel & Vulnerability",
            "text": "Threat actor Lazarus Group used CVE-2026-0001 on IP 192.168.1.1.",
            "lang": "en",
            "expected": [("threat_actor", "Lazarus Group"), ("vulnerability", "CVE-2026-0001"), ("digital_indicator", "192.168.1.1")]
        },
        {
            "name": "FinTech & Transport ID",
            "text": "Paid $12.5B for Flight SV205 via IBAN SA12345.",
            "lang": "en",
            "expected": [("money", "$12.5B"), ("transport_id", "SV205")]
        },
        {
            "name": "Temporal & Media Reclassification",
            "text": "גלובס reports Q3 2026 earnings are up.",
            "lang": "he",
            "expected": [("media", "גלובס"), ("date_time", "2026")]
        },
        {
            "name": "Project & Versioning",
            "text": "Deployment of Project-ORION/IL v3.1 is scheduled.",
            "lang": "en",
            "expected": [("project", "Project-ORION/IL v3.1")]
        }
    ]

    for case in test_cases:
        print(f"\n--- Testing: {case['name']} ---")
        print(f"Input: {case['text']}")
        try:
            result = await service.process_text(case["text"], language=case.get("lang"))
            print(f"Entities Found: {len(result.entities)}")
            
            found_map = {(ent.label, ent.text): True for ent in result.entities}
            
            for label, text in case.get("expected", []):
                if (label, text) in found_map:
                    print(f"  ✅ Found [{label}] '{text}'")
                else:
                    similar = [f"[{l}] '{t}'" for l, t in found_map.keys() if l == label or text in t]
                    print(f"  ❌ NOT FOUND: [{label}] '{text}'. Found: {similar}")

            for ent in result.entities:
                actual_span = case["text"][ent.start:ent.end]
                if actual_span != ent.text:
                    print(f"  ⚠️ OFFSET SHIFT: '{ent.text}' points to '{actual_span}' (Clitic stripped)")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_production_pipeline())
