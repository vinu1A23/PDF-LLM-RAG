
import uvicorn
from T13_subt5_rag_backend import *
from fastapi import FastAPI
from pydantic import BaseModel
from typing import  Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class query_data(BaseModel):
    name: str
    url: Optional[str] = None
    query: Optional[str] = None
    
   

def extract_text_from_pdf(pdf_content):
    docs = pdf_content
    #print("len of docs is" + str(len(docs)))
    all_data=""
    for i in range(len(docs)):
        all_data += docs[i].page_content[0:]
    return all_data


description = {'PDF LLM RAG':'A Web API for PDF LLM RAG'}

@app.get("/")
async def index():
    global description
    return description

@app.post("/download_pdf/")
async def get_pdf(data:query_data):
    url = data.url
    name = data.name
    print("reached till data.get(url)")
    if not url:
        return JSONResponse({"error": "URL is required"}), 400

    if not name:
        return JSONResponse({"error": "Name is required"}), 400

    try:
        pdf_await = await download_pdf(url,name)
        pdf_content = await load_pdf(name)
        text = extract_text_from_pdf(pdf_content)
        return JSONResponse({"text": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}), 500


@app.post("/preprocess_pdf/")
async def preprocess_pdf(data:query_data):
    name = data.name

    if not name:
        return JSONResponse({"error": "Name is required"}), 400

    try:
        pdf_content = await load_pdf(name)
        doc_splitted = await split_doc(pdf_content)
        return JSONResponse({"doc_splitted": str(doc_splitted)})
    except Exception as e:
        return JSONResponse({"error": str(e)}), 500

@app.post("/gen_context/")
async def gen_context(data:query_data):
    global embedding
    global embedding_loaded
    query = data.query
    name = data.name

    if not name:
        return JSONResponse({"error": "Name is required"}), 400
    if not query:
        return JSONResponse({"error": "Query is required"}), 400

    try:
        pdf_content = await load_pdf(name)
        doc_splitted = await split_doc(pdf_content)
        if embedding_loaded == False:
            embedding = load_embedding()
            embedding_loaded = True

        db = vector_database_setup(doc_splitted, embedding)
        context = generate_context(db, query)
        return JSONResponse({"context": context})
    except Exception as e:
        return JSONResponse({"error": str(e)}), 500

@app.post("/ans_query/")
async def ans_queryt(data:query_data):
    global embedding
    global embedding_loaded
    global model
    global model_loaded
    global tokenizer

    query = data.query
    name = data.name

    if not name:
        return JSONResponse({"error": "Name is required"}), 400
    if not query:
        return JSONResponse({"error": "Query is required"}), 400

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

        return JSONResponse({"answer": answer,"context":context,"preprocessed_pdf":str(doc_splitted),"content":extract_text_from_pdf(pdf_content)})
    except Exception as e:
        return JSONResponse({"error": str(e)}), 500






if __name__ == "__main__":
    uvicorn.run("T13_subt5_main:app", reload=True)
