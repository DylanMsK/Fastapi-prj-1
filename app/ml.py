import json
import numpy as np

from typing import Optional, List
from pathlib import Path
from dataclasses import dataclass

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .encoders import encode_to_json


@dataclass
class AI_MODEL:
    model_path: Path
    tokenizer_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    model = None
    tokenizer = None
    metadata = None

    def __post_init__(self):
        print(self.model_path)
        if self.model_path.exists():
            self.model = load_model(self.model_path)
        if self.tokenizer_path:
            if self.tokenizer_path.exists():
                if self.tokenizer_path.name.endswith("json"):
                    self.tokenizer = tokenizer_from_json(self.tokenizer_path.read_text())
        if self.metadata_path:
            if self.metadata_path.exists():
                if self.metadata_path.name.endswith("json"):
                    self.metadata = json.loads(self.metadata_path.read_text())

    def get_model(self):
        if not self.model:
            raise Exception("Model is not implemented")
        return self.model

    def get_tokenizer(self):
        if not self.tokenizer:
            raise Exception("Tokenizer is not impremented")
        return self.tokenizer

    def get_metadata(self):
        if not self.metadata:
            raise Exception("Metadata is not impremented")
        return self.metadata

    def get_sequences_from_text(self, texts: List[str]):
        tokenizer = self.get_tokenizer()
        sequences = tokenizer.texts_to_sequences(texts)
        return sequences

    def get_input_from_sequences(self, sequences):
        maxlen = self.metadata.get("max_sequence") or 280
        x_input = pad_sequences(sequences, maxlen=maxlen)
        return x_input
    
    def get_label_legend_inverted(self):
        legend = self.metadata.get("labels_legend_inverted") or {}
        if len(legend) != 2:
            raise Exception("Failed to loaing legend")
        return legend

    def get_label_pred(self, idx, val):
        legend = self.get_label_legend_inverted()
        return {"label": legend[str(idx)], "confidence": val}
    
    def get_top_pred_labeld(self, preds):
        top_idx_val = np.argmax(preds)
        return self.get_label_pred(top_idx_val, preds[top_idx_val])

    def predict_text(self, query: str, include_top: bool = True, to_json: bool = True):
        model = self.get_model()
        sequences = self.get_sequences_from_text([query])
        x_input = self.get_input_from_sequences(sequences)
        preds = model.predict(x_input)[0]
        labeled_preds = [self.get_label_pred(i, x) for i, x in enumerate(preds)]
        results = {
            "predictions": labeled_preds,
        }

        if include_top:
            results["top"] = self.get_top_pred_labeld(preds)

        if to_json:
            results = encode_to_json(results)

        return results