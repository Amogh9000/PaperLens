from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import engine
import db

app = FastAPI(title="PaperLens OMR API", version="1.0")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


def _load_key_for_set(key_set: str) -> dict:
    key_file = f"set_{key_set}.json"
    key_path = os.path.join(os.path.dirname(__file__), "maps", key_file)
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Key file not found: {key_file}")
    with open(key_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/evaluate")
async def evaluate(
    file: UploadFile = File(...),
    key_set: str = Form("A"),
):
    try:
        contents = await file.read()
        key_json = _load_key_for_set(key_set)
        res = engine.evaluate_omr(contents, key_json)

        # Persist
        try:
            db_id = db.save_result(
                student_id=res.student_id,
                subject_scores=res.subject_scores,
                total_score=res.total_score,
                evaluation_time_str=res.evaluation_time,
                confidence_score=res.confidence_score,
                key_set=key_set,
                source_file=file.filename,
            )
        except Exception as e:
            db_id = None

        overlay_b64 = base64.b64encode(res.annotated_image or b"").decode("ascii")
        return JSONResponse(
            content={
                "student_id": res.student_id,
                "subject_scores": res.subject_scores,
                "total_score": res.total_score,
                "evaluation_time": res.evaluation_time,
                "confidence_score": res.confidence_score,
                "db_id": db_id,
                "overlay_png_base64": overlay_b64,
            }
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


# Run with: uvicorn fastapi_app:app --reload
