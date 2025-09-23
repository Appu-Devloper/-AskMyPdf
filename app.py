from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import re
from datetime import timedelta
import requests
import json

# NEW: Google Generative AI
import google.generativeai as genai

# -------------------------
# Basic Flask setup
# -------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
app.permanent_session_lifetime = timedelta(hours=8)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTS = {"pdf"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory session cache (OK for dev; use Redis for prod)
PDF_CACHE = {}

# -------------------------
# Gemini setup
# -------------------------
genai.configure(api_key="AIzaSyCr9aLoO0RBI2kYbKriEwCU9ZetEfLWesg")
GEMINI_MODEL_ID = "gemini-2.0-flash"

# Optional tuning
GENERATION_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 1024,
}

# -------------------------
# Ollama setup
# -------------------------
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"  # Change this to your preferred model

# -------------------------
# Helpers
# -------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def extract_text_with_pdfplumber(pdf_path: str, max_pages: int | None = None) -> str:
    try:
        parts = []
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
            for page in pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
        return ("\n".join(parts)).strip()
    except Exception:
        return ""

def extract_text_with_ocr(pdf_path: str, max_pages: int | None = None) -> str:
    try:
        images = convert_from_path(pdf_path)
        images = images if max_pages is None else images[:max_pages]
        ocr = []
        for img in images:
            ocr.append(pytesseract.image_to_string(img))
        return ("\n".join(ocr)).strip()
    except Exception:
        return ""

def windowed_matches(text: str, query: str, window_chars: int = 200, max_hits: int = 4) -> list[str]:
    """Small context windows around keyword matches (case-insensitive)."""
    results = []
    if not query:
        return results
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    for m in pattern.finditer(text):
        s = max(0, m.start() - window_chars)
        e = min(len(text), m.end() + window_chars)
        results.append(text[s:e].strip())
        if len(results) >= max_hits:
            break
    return results

def build_context(text: str, question: str, hard_cap_chars: int = 5000) -> str:
    """
    Keep prompt small & relevant:
    1) try keyword windows
    2) else fall back to first N chars
    """
    hits = windowed_matches(text, question, window_chars=400, max_hits=10)
    if hits:
        ctx = "\n\n---\n\n".join(hits)
    else:
        ctx = text[:hard_cap_chars]
    return ctx

def query_ollama(question: str, context: str) -> str:
    """Query Ollama local model"""
    try:
        prompt = (
            "You are a helpful PDF Q&A assistant. "
            "Answer STRICTLY using the provided PDF context. "
            "If the answer is not present in the context, respond: "
            "\"I couldn't find this in the PDF.\" "
            "Prefer concise bullet points and include exact values/units when present.\n\n"
            f"PDF Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer from the context above:"
        )
        
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "max_tokens": 1024
                }
            },
            timeout=100
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return f"Ollama error: {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return "Ollama server not running. Please start Ollama on localhost:11434"
    except Exception as e:
        return f"Ollama error: {str(e)}"

SYSTEM_INSTRUCTION = (
    "You are a helpful PDF Q&A assistant. "
    "Answer STRICTLY using the provided PDF context. "
    "If the answer is not present in the context, respond: "
    "\"I couldn't find this in the PDF.\" "
    "Prefer concise bullet points and include exact values/units when present."
)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html") if os.path.exists("templates/index.html") else redirect(url_for("pdf_qa_page"))

@app.route("/app")
def pdf_qa_page():
    return render_template("pdf_qa.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["pdf"]
    if f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(f.filename):
        return jsonify({"error": "Only .pdf files are allowed"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)

    # Tip: set max_pages=2 for faster preview; use None for full extraction
    text = extract_text_with_pdfplumber(filepath, max_pages=None)
    if not text:
        text = extract_text_with_ocr(filepath, max_pages=2)

    session.permanent = True
    sid = session.get("_id") or os.urandom(8).hex()
    session["_id"] = sid
    PDF_CACHE[sid] = {"text": text, "path": filepath}

    return jsonify({"text": text})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    model_choice = data.get("model", "gemini")  # Default to Gemini

    if not question:
        return jsonify({"answer": "Please enter a question."})

    sid = session.get("_id")
    if not sid or sid not in PDF_CACHE or not PDF_CACHE[sid].get("text"):
        return jsonify({"answer": "No PDF uploaded yet."})

    text = PDF_CACHE[sid]["text"]

    # Build a compact, relevant context to save tokens
    context = build_context(text, question, hard_cap_chars=20000)

    try:
        if model_choice == "ollama":
            # Use Ollama local model
            answer = query_ollama(question, context)
        else:
            # Use Gemini (default)
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL_ID,
                generation_config=GENERATION_CONFIG,
                system_instruction=SYSTEM_INSTRUCTION,
            )

            prompt = (
                "PDF Context:\n"
                "----------------------------------------\n"
                f"{context}\n"
                "----------------------------------------\n\n"
                f"Question: {question}\n\n"
                "Answer from the context above."
            )

            resp = model.generate_content(prompt)
            answer = (resp.text or "").strip() if resp else ""

        if not answer:
            answer = "I couldn't find relevant information in the PDF."

        return jsonify({"answer": answer})

    except Exception as e:
        # Fail safely
        return jsonify({"answer": f"Error calling {model_choice}: {type(e).__name__}: {e}"}), 500

@app.route("/health")
def health_check():
    """Health check for Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return jsonify({
                "ollama_running": True,
                "available_models": [model["name"] for model in models],
                "status": "healthy"
            })
        else:
            return jsonify({
                "ollama_running": False,
                "status": "Ollama not responding"
            })
    except:
        return jsonify({
            "ollama_running": False,
            "status": "Cannot connect to Ollama"
        })

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Local dev server
    app.run(debug=True, host="0.0.0.0", port=5000)