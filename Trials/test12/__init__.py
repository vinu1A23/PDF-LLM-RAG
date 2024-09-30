from flask import Flask, request, jsonify
from flask_cors import CORS
from test12_llm_rag_backend_dockerize_ready import *


app = Flask(__name__)
CORS(app)


def extract_text_from_pdf(pdf_content):
    docs = pdf_content
    logger.info("len of docs is" + str(len(docs)))
    all_data=""
    for i in range(len(docs)):
        all_data += docs[i].page_content[0:]
    logger.info(f"Printing a bit of all_data {all_data[:100]}")
    return all_data


@app.route('/get_pdf', methods=['POST'])
async def get_pdf():
    data = request.json
    url = data.get('url')
    name = data.get('name')
    logger.debug("reached till data.get(url)")
    if not url:
        return jsonify({"error": "URL is required"}), 400

    if not name:
        return jsonify({"error": "Name is required"}), 400

    try:
        pdf_await = await download_pdf(url,name)
        pdf_content = await load_pdf(name)
        text = extract_text_from_pdf(pdf_content)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/preprocess_pdf', methods=['POST'])
async def preprocess_pdf():
    data = request.json
    """query = data.get('query')"""
    name = data.get('name')

    if not name:
        return jsonify({"error": "Name is required"}), 400

    try:
        pdf_content = await load_pdf(name)
        doc_splitted = await split_doc(pdf_content)
        return jsonify({"doc_splitted": str(doc_splitted)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/gen_context', methods=['POST'])
async def gen_context():
    global embedding
    global embedding_loaded

    data = request.json
    query = data.get('query')
    name = data.get('name')

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        pdf_content = await load_pdf(name)
        doc_splitted = await split_doc(pdf_content)
        if embedding_loaded == False:
            embedding = load_embedding()
            embedding_loaded = True

        db = vector_database_setup(doc_splitted, embedding)
        context = generate_context(db, query)
        return jsonify({"context": context})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ans_query', methods=['POST'])
async def ans_queryt():
    global embedding
    global embedding_loaded
    global model
    global model_loaded
    global tokenizer

    data = request.json
    query = data.get('query')
    name = data.get('name')

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        pdf_content = await load_pdf(name)
        doc_splitted = await split_doc(pdf_content)
        if embedding_loaded == False:
            embedding = load_embedding()
            embedding_loaded = True

        db = vector_database_setup(doc_splitted, embedding)
        context = generate_context(db, query)
        if model_loaded == False:
            model, tokenizer = load_model()
            model_loaded = True
        answer = answer_query(context, query, model, tokenizer)

        return jsonify({"answer": answer,"context":context,"preprocessed_pdf":str(doc_splitted),"content":extract_text_from_pdf(pdf_content)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

