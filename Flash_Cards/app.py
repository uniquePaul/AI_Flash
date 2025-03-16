from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import infer_auto_device_map
import json
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
TEXT_FOLDER = "extracted_text"
FLASHCARD_FOLDER = "cards"  # ✅ 新增文件夹存储 Flash Cards
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)
os.makedirs(FLASHCARD_FOLDER, exist_ok=True)  # ✅ 创建存储 Flash Cards 的文件夹

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TEXT_FOLDER"] = TEXT_FOLDER
app.config["FLASHCARD_FOLDER"] = FLASHCARD_FOLDER  # ✅ Flash Cards 存储路径

# ✅ 加载本地 DeepSeek-Qwen 模型
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# device_map = infer_auto_device_map(model)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map=device_map)
@app.route("/")
def home():
    return jsonify({"message": "Flask backend is running!"})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """上传 PDF，提取文本，并生成 Flash Cards"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    text_filename = os.path.splitext(file.filename)[0] + ".txt"
    text_file_path = os.path.join(app.config["TEXT_FOLDER"], text_filename)
    save_text_to_file(text_file_path, extracted_text)

    flash_cards = generate_flash_cards(extracted_text)
    extracted_text = extract_text_from_pdf(file_path)

    text_filename = os.path.splitext(file.filename)[0] + ".txt"
    text_file_path = os.path.join(app.config["TEXT_FOLDER"], text_filename)
    save_text_to_file(text_file_path, extracted_text)

    # flash_cards = generate_flash_cards(extracted_text)
    # print(flash_cards)
    return jsonify({
        "filename": file.filename,
        "pdf_filepath": file_path,
        "text_filepath": text_file_path,
        #"flashcard_filepath": flashcard_path    
        # "flash_cards": flash_cards
    })

@app.route("/flashcards", methods=["GET"])
def get_flashcards():
    """Retrieve Flash Cards JSON from the `cards/` folder."""
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Missing filename parameter"}), 400

    flashcard_filename = os.path.splitext(filename)[0] + "_flashcards.json"
    flashcard_path = os.path.join(app.config["FLASHCARD_FOLDER"], flashcard_filename)
    print(flashcard_path)
    if not os.path.exists(flashcard_path):
        return jsonify({"error": "Flash Cards not found for this file"}), 404
    with open(flashcard_path, "r", encoding="utf-8") as f:
        flash_cards = json.load(f)  # ✅ Load JSON as a Python dictionary

    return jsonify({"flash_cards": flash_cards})


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

# def generate_flash_cards(text):
#     """使用 DeepSeek-Qwen 生成 Flash Cards"""
#     prompt = f"""
#     你是一个 AI 学习助手，请根据以下文本生成 10 张 Flash Cards。
#     每张 Flash Card 包含：
#     - `"description"`: 知识点的详细描述
#     - `"abbreviation"`: 知识点的关键词

#     **以下是文本内容:**
#     {text[:1000]}  # 限制文本长度，避免输入过长导致错误

#     **请返回 JSON 格式的 Flash Cards**
#     """

#     device = "cpu"  # 强制使用 CPU（MPS 可能导致错误）
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)  # 限制 max_length
#     outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9)
#     response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     return response_text  # 直接返回 AI 生成的文本

def generate_flash_cards(text):
    """生成 Flash Cards (这里可以改成真实的 AI 生成)"""
    example_flashcards = [
        {"description": "What is AI?", "abbreviation": "Artificial Intelligence"},
        {"description": "What is Machine Learning?", "abbreviation": "ML"},
        {"description": "What is Deep Learning?", "abbreviation": "DL"},
    ]
    import json
    return json.dumps(example_flashcards, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
