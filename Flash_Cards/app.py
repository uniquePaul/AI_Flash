from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PDF 解析库 PyMuPDF

app = Flask(__name__)
CORS(app)  # 允许跨域请求

UPLOAD_FOLDER = "uploads"
TEXT_FOLDER = "extracted_text"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 确保上传目录存在
os.makedirs(TEXT_FOLDER, exist_ok=True)  # 确保文本存储目录存在
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEXT_FOLDER"] = TEXT_FOLDER

@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running!"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """上传 PDF，提取文本，并保存到 .txt 文件"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # 保存 PDF 文件
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # 提取 PDF 文本
    extracted_text = extract_text_from_pdf(file_path)

    # 保存文本到 .txt 文件
    text_filename = os.path.splitext(file.filename)[0] + ".txt"
    text_file_path = os.path.join(app.config["TEXT_FOLDER"], text_filename)
    save_text_to_file(text_file_path, extracted_text)

    return jsonify({
        "filename": file.filename,
        "pdf_filepath": file_path,
        "text_filepath": text_file_path,
        "message": "Text extracted and saved successfully!"
    })

def extract_text_from_pdf(pdf_path):
    """使用 PyMuPDF 解析 PDF 并提取文本"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"

    return text.strip() if text.strip() else "⚠️ No text found in PDF!"

def save_text_to_file(file_path, text):
    """将提取的文本保存到 .txt 文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
