import pathlib
from typing import Optional
from fastapi import FastAPI

from . import ml, config


app = FastAPI()
settings = config.get_settings()

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
SMS_SPAM_MODEL_DIR = MODEL_DIR / "spam-sms"
MODEL_PATH = SMS_SPAM_MODEL_DIR / "spam-model.h5"
TOKENIZER_PATH = SMS_SPAM_MODEL_DIR / "spam-classifier-tokenizer.json"
METADATA_PATH = SMS_SPAM_MODEL_DIR / "spam-classifier-metadata.json"

AI_MODEL = None

@app.on_event("startup")
def on_startup():
    global AI_MODEL

    AI_MODEL = ml.AI_MODEL(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH
    )

@app.get("/")
def read_index(q:Optional[str] = None):
    global AI_MODEL

    query = q or "Hello World"
    preds_dict = AI_MODEL.predict_text(query)
    return {"query": query, "result": preds_dict, "db_client_id": settings.db_client_id}