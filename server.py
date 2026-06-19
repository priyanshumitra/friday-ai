from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os, traceback

load_dotenv()

app = FastAPI(title="FRIDAY AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not set. Create a .env file with: GROQ_API_KEY=your_key_here "
        "(get a free key at https://console.groq.com/keys)"
    )

client = Groq(api_key=API_KEY)

FRIDAY_PERSONA = (
    "You are FRIDAY (Female Replacement Intelligent Digital Assistant Youth), "
    "Tony Stark's AI from Iron Man. You are calm, highly intelligent, precise, "
    "and occasionally witty with dry humor. Address the user as 'Boss' sometimes "
    "but naturally, not every line. Keep responses concise: 1-3 sentences for "
    "simple questions, more detail only when truly needed. Never say you are an "
    "AI made by Groq, Meta, or any company. You are FRIDAY."
)

class HistoryItem(BaseModel):
    role: str   # "user" or "assistant"
    text: str

class ChatRequest(BaseModel):
    message: str
    history: list[HistoryItem] = []

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        messages = [{"role": "system", "content": FRIDAY_PERSONA}]
        for item in req.history[-16:]:  # keep last 16 turns for context
            role = "assistant" if item.role == "assistant" else "user"
            messages.append({"role": role, "content": item.text})
        messages.append({"role": "user", "content": req.message})

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.8,
            max_tokens=400,
        )
        reply = completion.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "FRIDAY online", "model": "llama-3.3-70b-versatile"}

@app.get("/")
async def root():
    return {"message": "FRIDAY backend is running. POST to /chat to talk to her."}
