from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import aiohttp
from test11_llm_rag_backend import load_pdf,download_pdf
import logging

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)
logging.basicConfig(filename='Test1_tsub01_backend.log', encoding='utf-8', level=logging.DEBUG)

def extract_text_from_pdf(pdf_content):
    docs = pdf_content
    print("len of docs is" ,len(docs))
    all_data=""
    for i in range(len(docs)):
        all_data+=docs[i].page_content[0:]
    print("Printing a bit of all_data ",all_data[:100])
    return all_data


@app.route('/get_pdf', methods=['POST'])
async def get_pdf():
    data = request.json
    url = data.get('url')
    """name = data.get('name')"""
    logger.debug("reached till data.get(url)")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    """
    if not name:
        return jsonify({"error": "Name is required"}), 400
    """
    try:
        pdf_await = await download_pdf("https://arxiv.org/pdf/2107.03700","opencv.pdf")
        pdf_content = await load_pdf("opencv.pdf")
        text = extract_text_from_pdf(pdf_content)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

