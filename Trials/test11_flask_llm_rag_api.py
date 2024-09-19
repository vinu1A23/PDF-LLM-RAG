from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import aiohttp
import PyPDF2
from io import BytesIO

app = Flask(__name__)
CORS(app)

async def fetch_pdf(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def extract_text_from_pdf(pdf_content):
    pdf_file = BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


@app.route('/get_pdf', methods=['POST'])
async def get_pdf():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        pdf_content = await fetch_pdf(url)
        text = extract_text_from_pdf(pdf_content)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/enter_query', methods=['POST'])
async def enter_query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Here you would implement your query processing logic
    # For this example, we'll just return a dummy response
    response = {
        "answer": f"This is a dummy answer to the query: {query}",
        "confidence": 0.85,
        "source": "Dummy source"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
