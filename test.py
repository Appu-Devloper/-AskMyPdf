import tkinter as tk
from tkinter import filedialog
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import os
 
def select_pdf_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PDF file",
        filetypes=[("PDF files", "*.pdf")]
    )
    return file_path
 
def extract_text_with_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            return text.strip()
    except:
        return ''
 
def extract_text_with_ocr(pdf_path):
    print("🔍 Using OCR fallback...")
    images = convert_from_path(pdf_path)
    ocr_text = ''
    for image in images:
        ocr_text += pytesseract.image_to_string(image)
    return ocr_text.strip()
 
# === Main ===
pdf_file = select_pdf_file()
if pdf_file:
    print(f"\n✅ Selected: {pdf_file}")
    text = extract_text_with_pdfplumber(pdf_file)
    if not text:
        text = extract_text_with_ocr(pdf_file)
    
    if text:
        print("\n📄 Extracted Text:\n")
        print(text)
    else:
        print("⚠️ No text could be extracted from the PDF.")
else:
    print("❌ No file selected.")
 
 