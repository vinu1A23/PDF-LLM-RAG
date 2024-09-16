import os
import transformers
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer #Replace "your-model-name" with the actual name of your model

from langchain_community.document_loaders import PyPDFLoader

import urllib.request
import shutil

url= "https://arxiv.org/pdf/1906.08172"
file_name = "Mediapipe.pdf"

url= input("enter url of the pdf to download")
file_name= input("enter name of pdf(along with .pdf extension)")
with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)

model_name = os.getenv("MODEL_NAME")
print(model_name)
model_config_path = os.getenv("MODEL_CONFIG") #Load the model config


question_answerer = pipeline("question-answering", model='distilbert/distilbert-base-cased-distilled-squad')

if file_name is None:
    file_path = "Mediapipe.pdf" # put the pdf in the folder of the python file itself, we had mediapipe
else :
    file_path = file_name
print("file path given as", file_path)
loader = PyPDFLoader(file_path)

docs = loader.load()

print("len of docs is" ,len(docs))
all_data=""
for i in range(len(docs)):
    all_data+=docs[i].page_content[0:]
print("Printing a bit of all_data ",all_data[:100])

context = all_data
query = input("Enter the question")
result = question_answerer(question= query,     context=context)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
