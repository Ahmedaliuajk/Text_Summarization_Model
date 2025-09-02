from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import os
import torch
from transformers import pipeline

# Load environment variables (optional: create a .env file)
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/bart-large-cnn")  # change to distilbart for lighter model
API_KEY = os.getenv("API_KEY", "changeme")  # replace in .env for security

# Security header
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Request schema
class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 142
    min_length: int = 56
    num_beams: int = 4

# FastAPI app
app = FastAPI(title="Summarization API")

@app.on_event("startup")
async def startup_event():
    global summarizer
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=MODEL_NAME, device=device)

# API key validation
def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# Summarization endpoint
@app.post("/summarize", dependencies=[Depends(validate_api_key)])
async def summarize(req: SummarizeRequest):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Text too short")

    outputs = summarizer(
        req.text,
        max_length=req.max_length,
        min_length=req.min_length,
        num_beams=req.num_beams,
        truncation=True
    )
    return {"summary": outputs[0]["summary_text"]}
 


 #### s_Q7eW04ny53hsjo6Uqgpg