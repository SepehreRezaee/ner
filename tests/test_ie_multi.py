import asyncio
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.engines.nlp.ner.model import build_service

async def run_comprehensive_test():
    config_path = Path("configs/ner.yaml")
    print(f"Loading service from {config_path}...")
    service = build_service(config_path)
    
    # 1. Multi-lingual NER
    print("\n--- 1. Multi-lingual NER Test ---")
    texts = [
        ("Persian", "علی در تهران برای شرکت اپل کار می‌کند."),
        ("English", "John works for Apple Inc. in Cupertino."),
        ("Arabic", "أحمد يعمل في شركة أبل في كوبرتينو."),
        ("Hebrew", "דוד עובד בחברת אפל בקוברטינו.")
    ]
    labels = ["person", "organization", "location"]
    
    for lang, text in texts:
        print(f"\n[{lang}] Input: {text}")
        result = await service.process_text(text, labels=labels)
        for ent in result.entities:
            print(f"  - {ent.label}: {ent.text} (conf: {ent.score:.2f})")

    # 2. Text Classification
    print("\n--- 2. Text Classification Test (Sentiment) ---")
    sentiment_text = "این محصول عالی است اما کمی گران است." # This product is excellent but a bit expensive.
    sentiment_labels = {"sentiment": ["positive", "negative", "neutral"]}
    results = await service.classify(sentiment_text, sentiment_labels)
    print(f"Input: {sentiment_text}")
    print(f"Result: {json.dumps(results, ensure_ascii=False, indent=2)}")

    # 3. Structured Data Extraction (JSON)
    print("\n--- 3. Structured Data Extraction (JSON) Test ---")
    product_text = "آیفون ۱۵ پرو با ظرفیت ۲۵۶ گیگابایت و قیمت ۹۹۹ دلار."
    schema = {
        "product": [
            "name::str::Full product name",
            "storage::str::Storage capacity",
            "price::str::Price"
        ]
    }
    results = await service.extract_json(product_text, schema)
    print(f"Input: {product_text}")
    print(f"Result: {json.dumps(results, ensure_ascii=False, indent=2)}")

    # 4. Relation Extraction
    print("\n--- 4. Relation Extraction Test ---")
    rel_text = "Elon Musk founded SpaceX in 2002. SpaceX is located in Hawthorne."
    rel_labels = ["founded", "located_in"]
    results = await service.extract_relations(rel_text, rel_labels)
    print(f"Input: {rel_text}")
    print(f"Result: {json.dumps(results, ensure_ascii=False, indent=2)}")

    # 5. Expert IE Improvements Test (Normalization & Honorifics)
    print("\n--- 5. Expert IE Improvements Test ---")
    expert_texts = [
        ("Normalization", "ד״ר בשרונה-תל אביב."), # Normalization should handle ״ and -
        ("Honorific Merging", "ד\"ר יוסי כהן ביקר בבית החולים.") # ד"ר should merge with יוסי כהן
    ]
    for desc, text in expert_texts:
        print(f"\n[{desc}] Input: {text}")
        result = await service.process_text(text)
        for ent in result.entities:
            print(f"  - {ent.label}: {ent.text} (conf: {ent.score:.2f})")

    # 6. Universal Multilingual Pipeline Test
    print("\n--- 6. Universal Multilingual Pipeline Test ---")
    multilingual_tests = [
        ("Arabic Normalization", "أهلاً بك في مكة المكرمة، المملكة العربية السعودية.", "ar"),
        ("Persian Normalization", "آقای دکتر علی محمدی در دانشگاه تهران.", "fa"),
        ("English Regex Boost", "The total cost is $500 USD.", "en")
    ]
    for desc, text, lang in multilingual_tests:
        print(f"\n[{desc}] Input: {text}")
        result = await service.process_text(text, language=lang)
        for ent in result.entities:
            print(f"  - {ent.label}: {ent.text} (conf: {ent.score:.2f} source: {getattr(ent, 'source', 'model')})")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
