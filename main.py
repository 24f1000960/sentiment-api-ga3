from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load .env file
load_dotenv()

# Create FastAPI app
app = FastAPI()

# Point OpenAI client to AIPipe instead of OpenAI directly
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://aipipe.org/openai/v1"  # AIPipe proxy URL
)

# What the incoming request must look like
class CommentRequest(BaseModel):
    comment: str

# The strict schema we send to the AI to force structured output
sentiment_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "rating": {
                    "type": "integer",
                    "description": "1 = highly negative, 5 = highly positive"
                }
            },
            "required": ["sentiment", "rating"],
            "additionalProperties": False
        }
    }
}

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Analyze the sentiment of the comment. "
                        "Return sentiment as 'positive', 'negative', or 'neutral'. "
                        "Return a rating from 1 (highly negative) to 5 (highly positive)."
                    )
                },
                {
                    "role": "user",
                    "content": request.comment
                }
            ],
            response_format=sentiment_schema
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")