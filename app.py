from flask import Flask, render_template, request, jsonify
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

pdf_text_cache = ""

def extract_text_with_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except:
        return ""

def extract_text_with_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    ocr_text = ""
    for image in images:
        ocr_text += pytesseract.image_to_string(image)
    return ocr_text.strip()

@app.route("/")
def home():
    return render_template("index.html")  # onboarding slideshow

@app.route("/app")
def pdf_qa_page():
    return render_template("pdf_qa.html")  # main PDF Q&A UI

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global pdf_text_cache
    file = request.files["pdf"]
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = extract_text_with_pdfplumber(filepath)
        if not text:
            text = extract_text_with_ocr(filepath)

        pdf_text_cache = text
        return jsonify({"text": text})
    return jsonify({"error": "No file uploaded"}), 400

@app.route("/ask", methods=["POST"])
def ask_question():
    global pdf_text_cache
    data = request.json
    question = data.get("question", "").lower()

    if not pdf_text_cache:
        return jsonify({"answer": "No PDF uploaded yet."})

    lines = [line for line in pdf_text_cache.split("\n") if question in line.lower()]
    if lines:
        return jsonify({"answer": "\n".join(lines[:5])})
    else:
        return jsonify({"answer": "I couldn't find relevant information in the PDF."})

if __name__ == "__main__":
    app.run(debug=True)
