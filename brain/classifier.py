import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("jarvis.brain.classifier")


class IntentClassifier:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        intents_path: str = "brain/intents.yaml",
        threshold: float = 0.5,
    ) -> None:
        self._threshold = threshold
        self._intent_map: list[dict[str, Any]] = []

        logger.info("Cargando modelo de clasificacion: %s", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")

        raw = yaml.safe_load(Path(intents_path).read_text(encoding="utf-8"))
        intents = raw.get("intents", {}) if isinstance(raw, dict) else {}

        all_examples: list[str] = []
        for intent_name, intent_data in intents.items():
            examples = intent_data.get("examples", [])
            for ex in examples:
                all_examples.append(ex)
                self._intent_map.append({
                    "intent": intent_name,
                    "skill": intent_data.get("skill", ""),
                    "operation": intent_data.get("operation", ""),
                    "text": ex,
                })

        logger.info("Codificando %d ejemplos de entrenamiento...", len(all_examples))
        self._embeddings = self._model.encode(all_examples, normalize_embeddings=True)
        logger.info("Clasificador listo. %d intents, %d ejemplos.", len(intents), len(all_examples))

    def classify(self, text: str) -> dict[str, Any] | None:
        emb = self._model.encode([text], normalize_embeddings=True)
        similarities = np.dot(self._embeddings, emb[0])
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < self._threshold:
            return None

        entry = self._intent_map[best_idx]
        return {
            "intent": entry["intent"],
            "skill": entry["skill"],
            "operation": entry["operation"],
            "score": best_score,
        }
