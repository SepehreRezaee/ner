
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from src.engines.nlp.ner.gliner_model import GlinerModel, NerConfig
from src.engines.nlp.ner.model import NerService

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test():
    print("Initializing Config...")
    config = NerConfig(
        model_name="fastino/gliner2-multi-v1",
        labels=["person", "location", "organization"],
        cache_dir=Path(".cache/ner")
    )
    
    print("Initializing Model...")
    model = GlinerModel(config)
    
    print("Loading Model...")
    await model.load()
    
    service = NerService(config, model)
    
    sentences = [
        ("Persian", "علی در تهران زندگی می‌کند."),
        ("English", "John lives in New York."),
        ("Arabic", "أحمد يعيش في القاهرة."),
        ("Hebrew", "דוד גר בירושלים.")  # David lives in Jerusalem
    ]
    
    for lang, text in sentences:
        print(f"\n--- Testing {lang} ---")
        result = await service.process_text(text)
        for entity in result.entities:
            print(f"  {entity.label}: {entity.text} (score: {entity.score:.2f})")

if __name__ == "__main__":
    asyncio.run(test())
